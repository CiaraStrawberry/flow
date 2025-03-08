import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple

from tqdm import tqdm
from safetensors.torch import safe_open, save_file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchastic import Compass, StochasticAccumulator
import random

from transformers import T5Tokenizer
import wandb

# Import the LatentCache from our new module
from cache_utils import LatentCache

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.sampling import get_noise, get_schedule, denoise_cfg
from src.models.chroma.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
import src.lora_and_quant as lora_and_quant

from huggingface_hub import HfApi, upload_file
import time

@dataclass
class TrainingConfig:
    master_seed: int
    cache_minibatch: int
    train_minibatch: int
    offload_param_count: int
    lr: float
    weight_decay: float
    warmup_steps: int
    change_layer_every: int
    trained_single_blocks: int
    trained_double_blocks: int
    save_every: int
    save_folder: str
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_entity: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class InferenceConfig:
    inference_every: int
    inference_folder: str
    steps: int
    guidance: int
    cfg: int
    prompts: list[str]
    first_n_steps_wo_cfg: int
    image_dim: tuple[int, int]
    t5_max_length: int


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    shuffle_tags: bool
    tag_drop_percentage: float
    uncond_percentage: float
    resolution_step: int
    num_workers: int
    prefetch_factor: int
    ratio_cutoff: float
    thread_per_worker: int


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def create_distribution(num_points, device=None):
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


# Upload the model to Hugging Face Hub
def upload_to_hf(model_filename, path_in_repo, repo_id, token, max_retries=3):
    api = HfApi()

    for attempt in range(max_retries):
        try:
            upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"Model uploaded to {repo_id}/{path_in_repo}")
            return  # Exit function if successful

        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    print("Upload failed after multiple attempts.")


def sample_from_distribution(x, probabilities, num_samples, device=None):
    # Step 1: Compute the cumulative distribution function
    cdf = torch.cumsum(probabilities, dim=0)

    # Step 2: Generate uniform random samples
    uniform_samples = torch.rand(num_samples, device=device)

    # Step 3: Map uniform samples to the x values using the CDF
    indices = torch.searchsorted(cdf, uniform_samples, right=True)

    # Get the corresponding x values for the sampled indices
    sampled_values = x[indices]

    return sampled_values


def prepare_sot_pairings(latents):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    image_pos_id = prepare_latent_image_ids(n, h, w)

    # randomize ode timesteps
    # input_timestep = torch.round(
    #     F.sigmoid(torch.randn((n,), device=latents.device)), decimals=3
    # )
    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=latents.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=latents.device
    )

    timesteps = input_timestep[:, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = latents * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    target = noise - latents

    return noisy_latents, target, input_timestep, image_pos_id, latent_shape


def init_optimizer(model, trained_layer_keywords, lr, wd, warmup_steps):
    # TODO: pack this into a function
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append((name, param))
        else:
            param.requires_grad = False  # Optionally disable grad for others
    # return hooks so it can be released later on
    hooks = StochasticAccumulator.assign_hooks(model)
    # init optimizer
    optimizer = Compass(
        [
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" not in name and "norm" not in name)
                ]
            },
            {
                "params": [
                    param
                    for name, param in trained_params
                    if ("bias" in name or "norm" in name)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    return optimizer, scheduler, hooks, trained_params


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def optimizer_state_to(optimizer, device):
    for param, state in optimizer.state.items():
        for key, value in state.items():
            # Check if the item is a tensor
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def save_part(model, trained_layer_keywords, counter, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(
        filtered_state_dict, os.path.join(save_folder, f"trained_part_{counter}.pth")
    )


def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)


def save_config_to_json(filepath: str, **configs):
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def inference_wrapper(
    model,
    ae,
    t5_tokenizer,
    t5,
    seed: int,
    steps: int,
    guidance: int,
    cfg: int,
    prompts: list,
    rank: int,
    first_n_steps_wo_cfg: int,
    image_dim=(512, 512),
    t5_max_length=512,
):
    #############################################################################
    # test inference
    # aliasing
    SEED = seed
    WIDTH = image_dim[0]
    HEIGHT = image_dim[1]
    STEPS = steps
    GUIDANCE = guidance
    CFG = cfg
    FIRST_N_STEPS_WITHOUT_CFG = first_n_steps_wo_cfg
    DEVICE = model.device
    PROMPT = prompts

    T5_MAX_LENGTH = t5_max_length

    # store device state of each model
    t5_device = t5.device
    ae_device = ae.device
    model_device = model.device
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # init random noise
            noise = get_noise(len(PROMPT), HEIGHT, WIDTH, DEVICE, torch.bfloat16, SEED)
            noise, shape = vae_flatten(noise)
            noise = noise.to(rank)
            n, c, h, w = shape
            image_pos_id = prepare_latent_image_ids(n, h, w).to(rank)

            timesteps = get_schedule(STEPS, noise.shape[1])

            model.to("cpu")
            ae.to("cpu")
            t5.to(rank)  # load t5 to gpu
            text_inputs = t5_tokenizer(
                PROMPT,
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)

            t5_embed = t5(text_inputs.input_ids, text_inputs.attention_mask).to(rank)

            text_inputs_neg = t5_tokenizer(
                [""],
                padding="max_length",
                max_length=T5_MAX_LENGTH,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).to(t5.device)

            t5_embed_neg = t5(text_inputs_neg.input_ids, text_inputs.attention_mask).to(
                rank
            )

            text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)
            neg_text_ids = torch.zeros((len(PROMPT), T5_MAX_LENGTH, 3), device=rank)

            ae.to("cpu")
            t5.to("cpu")
            model.to(rank)  # load model to gpu
            latent_cfg = denoise_cfg(
                model,
                noise,
                image_pos_id,
                t5_embed,
                t5_embed_neg,
                text_ids,
                neg_text_ids,
                text_inputs.attention_mask,
                text_inputs_neg.attention_mask,
                timesteps,
                GUIDANCE,
                CFG,
                FIRST_N_STEPS_WITHOUT_CFG,
            )

            model.to("cpu")
            t5.to("cpu")
            ae.to(rank)  # load ae to gpu
            output_image = ae.decode(vae_unflatten(latent_cfg, shape))

            # restore back state
            model.to("cpu")
            t5.to("cpu")
            ae.to("cpu")

    return output_image


# Function to precompute and cache all data
def precompute_all_caches(dataset, dataloader, t5_tokenizer, t5, ae, model_config, 
                          dataloader_config, training_config, rank, world_size,
                          use_disk=True, cache_dir="cache"):
    """
    Precompute and cache all latents and text embeddings before training
    Returns a dataset map with keys for each item
    
    Args:
        use_disk: If True, store cache on disk instead of in memory
        cache_dir: Base directory for cache storage
    """
    print(f"Rank {rank}: Pre-computing and caching all data...")
    
    # Create cache directories based on rank to avoid conflicts in distributed training
    latent_cache_dir = f"{cache_dir}/latents_rank{rank}"
    text_cache_dir = f"{cache_dir}/text_rank{rank}"
    
    print(f"Rank {rank}: Using {'disk' if use_disk else 'RAM'} for caching")
    
    # Create caches with specified storage type
    latent_cache = LatentCache(use_disk=use_disk, cache_dir=latent_cache_dir, clear_existing=True)
    text_cache = LatentCache(use_disk=use_disk, cache_dir=text_cache_dir, clear_existing=True)
    
    # Store dataset_map for quick retrieval during training
    dataset_map = []
    
    # Load models to GPU once and keep them there during the entire caching process
    print(f"Rank {rank}: Loading models to GPU for caching...")
    t5_device_original = t5.device
    ae_device_original = ae.device
    
    # Move models to GPU
    t5 = t5.to(rank)
    ae = ae.to(rank)
    
    try:
        # Process all data in the dataloader
        for data_idx, data in tqdm(
            enumerate(dataloader),
            total=len(dataset),
            desc=f"Caching data, Rank {rank}",
            position=rank,
        ):
            images, caption, index = data[0]
            # Handle potential None captions
            caption = [x if x is not None else "" for x in caption]
            
            for mb_i in tqdm(
                range(
                    dataloader_config.batch_size
                    // training_config.cache_minibatch
                    // world_size
                ),
                desc=f"Caching batch {data_idx}, Rank {rank}",
                disable=data_idx > 0,  # Only show for first batch
            ):
                start_idx = mb_i * training_config.cache_minibatch
                end_idx = start_idx + training_config.cache_minibatch
                
                # Get current batch data
                current_captions = caption[start_idx:end_idx]
                current_images = images[start_idx:end_idx]
                
                # Create batch mapping entry
                batch_map = {
                    "data_idx": data_idx,
                    "mb_i": mb_i,
                    "image_keys": [],
                    "caption_keys": [],
                }
                
                # Generate keys for captions
                caption_keys = [
                    f"{latent_cache.hash_content(caption_text)}_rank{rank}" 
                    for caption_text in current_captions
                ]
                batch_map["caption_keys"] = caption_keys
                
                # Process and cache text embeddings
                texts_to_process = []
                text_indices = []
                
                for i, (caption_key, caption_text) in enumerate(zip(caption_keys, current_captions)):
                    cached_result = text_cache.get(caption_key)
                    if cached_result is None:
                        texts_to_process.append(caption_text)
                        text_indices.append(i)
                
                if texts_to_process:
                    # Process text with t5 (already on GPU)
                    text_inputs = t5_tokenizer(
                        texts_to_process,
                        padding="max_length",
                        max_length=model_config.t5_max_length,
                        truncation=True,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_tensors="pt",
                    ).to(t5.device)
                    
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        new_t5_embeds = t5(text_inputs.input_ids, text_inputs.attention_mask).to(
                            "cpu", non_blocking=True
                        )
                        new_masks = text_inputs.attention_mask.to("cpu", non_blocking=True)
                    
                    # Store in cache
                    for idx, orig_idx in enumerate(text_indices):
                        embed = new_t5_embeds[idx:idx+1]  # Keep batch dimension
                        mask = new_masks[idx:idx+1]  # Keep batch dimension
                        text_cache.put(caption_keys[orig_idx], (embed, mask))
                
                # Generate keys for images and process
                for i, img in enumerate(current_images):
                    # Create content hash based on image
                    if img.dim() >= 3:
                        img_shape = "_".join(str(dim) for dim in img.shape)
                        h, w = img.shape[-2], img.shape[-1]
                        center_h, center_w = h // 2, w // 2
                        patch_size = min(10, h // 2, w // 2)
                        h_start, h_end = center_h - patch_size, center_h + patch_size
                        w_start, w_end = center_w - patch_size, center_w + patch_size
                        img_sample = img[:, h_start:h_end, w_start:w_end]
                    else:
                        img_shape = "_".join(str(dim) for dim in img.shape)
                        img_sample = img
                    
                    # Create image key
                    content_hash = latent_cache.hash_content(img_sample)
                    img_key = f"img_{content_hash}_input_shape_{img_shape}_rank{rank}"
                    batch_map["image_keys"].append(img_key)
                    
                    # Check cache
                    if latent_cache.get(img_key) is None:
                        # Cache miss - need to process this image (ae already on GPU)
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            img_tensor = img.unsqueeze(0).to(rank)  # Add batch dimension
                            latent = ae.encode_for_train(img_tensor).to("cpu", non_blocking=True)
                        
                        # Store in cache
                        latent_cache.put(img_key, latent)
                
                # Add the batch mapping to our dataset map
                dataset_map.append(batch_map)
                
                # Log cache stats occasionally
                if (data_idx % 10 == 0 and mb_i == 0) or (data_idx == 0 and mb_i == 0):
                    if rank == 0:  # Only log from main process
                        print(f"Text cache stats: {text_cache.get_stats()}")
                        print(f"Latent cache stats: {latent_cache.get_stats()}")
                
                # Clear CUDA cache periodically to prevent OOM
                if mb_i % 10 == 0:
                    torch.cuda.empty_cache()
    
    finally:
        # Always move models back to their original devices when done
        print(f"Rank {rank}: Moving models back to original devices...")
        t5 = t5.to(t5_device_original)
        ae = ae.to(ae_device_original)
        torch.cuda.empty_cache()
    
    # Final stats
    if rank == 0:
        print(f"Final cache statistics:")
        print(f"Text cache: {text_cache.get_stats()}")
        print(f"Latent cache: {latent_cache.get_stats()}")
    
    return dataset_map, latent_cache, text_cache


def train_chroma(rank, world_size, args=None, debug=False):
    # Set default args if not provided
    if args is None:
        from types import SimpleNamespace
        args = SimpleNamespace(ram_cache=False, cache_dir='cache', debug=debug)

    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config.json")

    training_config = TrainingConfig(**config_data["training"])
    inference_config = InferenceConfig(**config_data["inference"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    extra_inference_config = [
        InferenceConfig(**conf) for conf in config_data["extra_inference_config"]
    ]

    # wandb logging
    if training_config.wandb_project is not None and rank == 0:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if training_config.wandb_key:
            wandb.login(key=training_config.wandb_key)
        wandb.init(
            project=training_config.wandb_project,
            name=f"{training_config.wandb_run}_{current_datetime}",
            entity=training_config.wandb_entity,
        )

    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )

    os.makedirs(inference_config.inference_folder, exist_ok=True)
    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        chroma_params._use_compiled = True
        with torch.device("meta"):
            model = Chroma(chroma_params)
        model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)

        # randomly train inner layers at a time
        trained_double_blocks = list(range(len(model.double_blocks)))
        trained_single_blocks = list(range(len(model.single_blocks)))
        random.shuffle(trained_double_blocks)
        random.shuffle(trained_single_blocks)
        # lazy :P
        trained_double_blocks = trained_double_blocks * 1000000
        trained_single_blocks = trained_single_blocks * 1000000

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(
            replace_keys(load_file_multipart(model_config.t5_path)), assign=True
        )
        t5.eval()
        t5.to(torch.bfloat16)
        if model_config.t5_to_8bit:
            cast_linear(t5, torch.float8_e4m3fn)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        # don't use this tag implication pruning it's slow!
        # preprocess the jsonl tags before training!
        # tag_implication_path="tag_implications.csv",
        base_res=dataloader_config.base_resolution,
        shuffle_tags=dataloader_config.shuffle_tags,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
    )

    # Create the dataloader ONCE, outside the epoch loop
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # batch size is handled in the dataset
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        prefetch_factor=dataloader_config.prefetch_factor,
        pin_memory=True,
        collate_fn=dataset.dummy_collate_fn,
    )
    
    # PRE-COMPUTE ALL CACHES BEFORE TRAINING - ONCE FOR ALL EPOCHS
    print(f"[Rank {rank}] Pre-computing all caches before training...")
    
    # Define cache directories based on command line args
    cache_base_dir = args.cache_dir
    
    # Update precompute_all_caches function to accept use_disk parameter
    dataset_map, latent_cache, text_cache = precompute_all_caches(
        dataset, dataloader, t5_tokenizer, t5, ae, 
        model_config, dataloader_config, training_config, rank, world_size,
        use_disk=not args.ram_cache,
        cache_dir=cache_base_dir
    )
    print(f"[Rank {rank}] Caching complete. Starting training loop...")
    
    if not debug:
        dist.barrier()  # Ensure all nodes have completed caching

    optimizer = None
    scheduler = None
    hooks = []
    optimizer_counter = 0
    epoch_counter = 0
    
    # Infinite training loop - keep the while True structure as requested
    while True:
        epoch_counter += 1
        training_config.master_seed += 1
        torch.manual_seed(training_config.master_seed)
        
        print(f"[Rank {rank}] Starting epoch {epoch_counter}")

        # TRAINING LOOP USING CACHED DATA
        for counter, batch_map in tqdm(
            enumerate(dataset_map),
            total=len(dataset_map),
            desc=f"Training epoch {epoch_counter}, Rank {rank}",
            position=rank,
        ):
            if counter % training_config.change_layer_every == 0:
                # periodically remove the optimizer and swap it with new one
                # aliasing to make it cleaner
                o_c = optimizer_counter
                n_ls = training_config.trained_single_blocks
                n_ld = training_config.trained_double_blocks
                trained_layer_keywords = (
                    [
                        f"double_blocks.{x}."
                        for x in trained_double_blocks[o_c * n_ld : o_c * n_ld + n_ld]
                    ]
                    + [
                        f"single_blocks.{x}."
                        for x in trained_single_blocks[o_c * n_ls : o_c * n_ls + n_ls]
                    ]
                    + ["txt_in", "img_in", "final_layer"]
                )

                # remove hooks and load the new hooks
                if len(hooks) != 0:
                    hooks = [hook.remove() for hook in hooks]

                optimizer, scheduler, hooks, trained_params = init_optimizer(
                    model,
                    trained_layer_keywords,
                    training_config.lr,
                    training_config.weight_decay,
                    training_config.warmup_steps,
                )

                optimizer_counter += 1

            # Retrieve cached data for this batch
            acc_latents = []
            acc_embeddings = []
            acc_mask = []
            
            for img_key, caption_key in zip(batch_map["image_keys"], batch_map["caption_keys"]):
                # Get image latent from cache
                latent = latent_cache.get(img_key)
                if latent is not None:
                    acc_latents.append(latent)
                
                # Get text embedding from cache
                text_data = text_cache.get(caption_key)
                if text_data is not None:
                    embed, mask = text_data
                    acc_embeddings.append(embed)
                    acc_mask.append(mask)
            
            # Only proceed if we have all data
            if not acc_latents or not acc_embeddings:
                print(f"Warning: Missing cached data for batch {counter}")
                continue
                
            # Concatenate batch data
            acc_latents = torch.cat(acc_latents, dim=0)
            acc_embeddings = torch.cat(acc_embeddings, dim=0)
            acc_mask = torch.cat(acc_mask, dim=0)
            
            # Move model to device
            model.to(rank)
            
            # Prepare for training
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # prepare flat image and the target lerp
                (
                    noisy_latents,
                    target,
                    input_timestep,
                    image_pos_id,
                    latent_shape,
                ) = prepare_sot_pairings(acc_latents.to(rank))
                noisy_latents = noisy_latents.to(torch.bfloat16)
                target = target.to(torch.bfloat16)
                input_timestep = input_timestep.to(torch.bfloat16)
                image_pos_id = image_pos_id.to(rank)

                # t5 text id for the model
                text_ids = torch.zeros((noisy_latents.shape[0], 512, 3), device=rank)
                # NOTE:
                # using static guidance 1 for now
                # this should be disabled later on !
                static_guidance = torch.tensor([0.0] * acc_latents.shape[0], device=rank)

            # Set inputs to requires_grad for backprop
            noisy_latents.requires_grad_(True)
            acc_embeddings.requires_grad_(True)

            # Process in mini-batches for better memory efficiency
            mb = training_config.train_minibatch
            loss_log = []
            
            for tmb_i in tqdm(
                range(dataloader_config.batch_size // mb // world_size),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                # Process each mini-batch separately
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        img=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_mask=acc_mask[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                    )
                    
                    # Calculate loss for this mini-batch
                    loss = (
                        F.mse_loss(
                            pred,
                            target[tmb_i * mb : tmb_i * mb + mb],
                        )
                        / dataloader_config.batch_size
                    )
                
                # Backward pass for this mini-batch
                torch.cuda.empty_cache()
                loss.backward()
                loss_log.append(loss.detach().clone() * dataloader_config.batch_size)
            
            # Calculate average loss across all mini-batches
            loss_log = sum(loss_log) / len(loss_log)
            
            # Free memory
            del acc_embeddings, noisy_latents, acc_latents
            torch.cuda.empty_cache()
            
            # Offload non-trainable params to save memory
            offload_param_count = 0
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in trained_layer_keywords):
                    if offload_param_count < training_config.offload_param_count:
                        offload_param_count += param.numel()
                        param.data = param.data.to("cpu", non_blocking=True)
                        
            optimizer_state_to(optimizer, rank)
            StochasticAccumulator.reassign_grad_buffer(model)

            if not debug:
                synchronize_gradients(model)

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            if training_config.wandb_project is not None and rank == 0:
                wandb.log({"loss": loss_log, "lr": training_config.lr, "epoch": epoch_counter})

            optimizer_state_to(optimizer, "cpu")
            torch.cuda.empty_cache()

            # Save model periodically
            if (counter + 1) % training_config.save_every == 0 and rank == 0:
                model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
                torch.save(
                    model.state_dict(),
                    model_filename,
                )
                if training_config.hf_token:
                    upload_to_hf(
                        model_filename,
                        model_filename,
                        training_config.hf_repo_id,
                        training_config.hf_token,
                    )
                    
            if not debug:
                dist.barrier()

            # Run inference periodically
            if (counter + 1) % inference_config.inference_every == 0:
                # Inference code goes here, unchanged from original
                all_grids = []

                # Get the data for the current batch to use in inference
                data_idx = batch_map["data_idx"]
                mb_i = batch_map["mb_i"]
                
                # Collect actual captions from the dataloader for the current batch
                batch_data = dataset[data_idx * world_size + rank]
                images, captions, _ = batch_data
                current_captions = captions[mb_i * training_config.cache_minibatch:][:1]
                
                preview_prompts = inference_config.prompts + current_captions

                for prompt in preview_prompts:
                    images_tensor = inference_wrapper(
                        model=model,
                        ae=ae,
                        t5_tokenizer=t5_tokenizer,
                        t5=t5,
                        seed=training_config.master_seed + rank,
                        steps=inference_config.steps,
                        guidance=inference_config.guidance,
                        cfg=inference_config.cfg,
                        prompts=[prompt],  # Pass single prompt as a list
                        rank=rank,
                        first_n_steps_wo_cfg=inference_config.first_n_steps_wo_cfg,
                        image_dim=inference_config.image_dim,
                        t5_max_length=inference_config.t5_max_length,
                    )

                    # gather from all gpus
                    if not debug:
                        gather_list = (
                            [torch.empty_like(images_tensor) for _ in range(world_size)]
                            if rank == 0
                            else None
                        )
                        dist.gather(images_tensor, gather_list=gather_list, dst=0)

                    if rank == 0:
                        # Concatenate gathered tensors
                        if not debug:
                            gathered_images = torch.cat(
                                gather_list, dim=0
                            )  # (total_images, C, H, W)
                        else:
                            gathered_images = images_tensor

                        # Create a grid for this prompt
                        grid = make_grid(
                            gathered_images.clamp(-1, 1).add(1).div(2),
                            nrow=8,
                            normalize=True,
                        )  # Adjust nrow as needed
                        all_grids.append(grid)

                for extra_inference in extra_inference_config:
                    for prompt in preview_prompts:
                        images_tensor = inference_wrapper(
                            model=model,
                            ae=ae,
                            t5_tokenizer=t5_tokenizer,
                            t5=t5,
                            seed=training_config.master_seed + rank,
                            steps=extra_inference.steps,
                            guidance=extra_inference.guidance,
                            cfg=extra_inference.cfg,
                            prompts=[prompt],  # Pass single prompt as a list
                            rank=rank,
                            first_n_steps_wo_cfg=extra_inference.first_n_steps_wo_cfg,
                            image_dim=extra_inference.image_dim,
                            t5_max_length=extra_inference.t5_max_length,
                        )

                        # gather from all gpus
                        if not debug:
                            gather_list = (
                                [
                                    torch.empty_like(images_tensor)
                                    for _ in range(world_size)
                                ]
                                if rank == 0
                                else None
                            )
                            dist.gather(images_tensor, gather_list=gather_list, dst=0)

                        if rank == 0:
                            # Concatenate gathered tensors
                            if not debug:
                                gathered_images = torch.cat(
                                    gather_list, dim=0
                                )  # (total_images, C, H, W)
                            else:
                                gathered_images = images_tensor

                            # Create a grid for this prompt
                            grid = make_grid(
                                gathered_images.clamp(-1, 1).add(1).div(2),
                                nrow=8,
                                normalize=True,
                            )  # Adjust nrow as needed
                            all_grids.append(grid)

                # send prompt to rank 0
                if rank != 0:
                    dist.send_object_list(caption[:1], dst=0)

                else:
                    all_prompt = []
                    # Rank 0 receives from all other ranks
                    for src_rank in range(1, world_size):
                        # Initialize empty list with the same size to receive strings
                        received_strings = [None]
                        # Receive the list of strings
                        dist.recv_object_list(received_strings, src=src_rank)
                        all_prompt.extend(received_strings)

                if rank == 0:
                    # Combine all grids vertically
                    final_grid = torch.cat(
                        all_grids, dim=1
                    )  # Concatenate along height dimension

                    # Save the combined grid
                    file_path = os.path.join(
                        inference_config.inference_folder, f"epoch{epoch_counter}_batch{counter}.jpg"
                    )
                    save_image(final_grid, file_path)
                    print(f"Combined image grid saved to {file_path}")

                    # upload preview to wandb
                    if training_config.wandb_project is not None:
                        wandb.log(
                            {
                                "example_image": wandb.Image(
                                    file_path,
                                    caption="\n".join(preview_prompts + all_prompt),
                                )
                            }
                        )

        # Save epoch checkpoint
        if rank == 0:
            print(f"Completed epoch {epoch_counter}")
            model_filename = f"{training_config.save_folder}/epoch_{epoch_counter}.pth"
            torch.save(
                model.state_dict(),
                model_filename,
            )
            if training_config.hf_token:
                upload_to_hf(
                    model_filename,
                    f"epoch_{epoch_counter}.pth",
                    training_config.hf_repo_id,
                    training_config.hf_token,
                )

    if not debug:
        dist.destroy_process_group()
