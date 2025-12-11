# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs distributed reconstructions with a pre-trained stage-1 model.
Inputs are loaded from an ImageFolder dataset or a parquet file, processed with center crops,
and the reconstructed images are saved as .png files alongside a packed .npz.
"""
import argparse
import math
import os
import sys
import shutil
from typing import List
from io import BytesIO
from pathlib import Path
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import daft

from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def center_crop_arr(pil_image: Image.Image, image_size: int):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def decode_img(bytes_data):
    """Decode image bytes to numpy array."""
    with Image.open(BytesIO(bytes_data)) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


class ParquetDataset(Dataset):
    """Dataset that loads images from a parquet file."""

    def __init__(self, parquet_path: str, pixel_key: str, transform=None):
        """
        Args:
            parquet_path: Path to the parquet file
            pixel_key: Key in the parquet file containing image bytes (e.g., "observations.images.front_img_1")
            transform: Optional transform to apply to images
        """
        self.parquet_path = parquet_path
        self.pixel_key = pixel_key
        self.transform = transform
        
        # Load parquet file and sort by index
        df = daft.read_parquet(str(parquet_path))
        df = df.sort("index")
        self.df = df
        
        # Convert to torch map dataset for efficient access
        self.dataset = df.to_torch_map_dataset()
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get image bytes from parquet
        img_dict = self.dataset[index][self.pixel_key]
        img_array = decode_img(img_dict['bytes'])  # (H, W, C) uint8
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        
        # Apply transform (includes center crop and ToTensor)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, index


def sanitize_component(component: str):
    """Replace OS separators to keep path components valid."""
    return component.replace(os.sep, "-")


def create_video_from_temp_frames(sample_dir: str, output_path: str, num_frames: int, fps: int = 30):
    """
    Create a side-by-side video from temporary frame files collected from all ranks.
    
    Args:
        sample_dir: Directory containing temp_frames_rank_* subdirectories
        output_path: Path to save the output video
        num_frames: Number of frames to include in the video
        fps: Frames per second for the video
    """
    frames = []
    
    for idx in tqdm(range(num_frames), desc="Loading frames for video"):
        # Search for frame in all rank temp directories
        frame_data = None
        for temp_dir in os.listdir(sample_dir):
            if temp_dir.startswith("temp_frames_rank_"):
                frame_path = os.path.join(sample_dir, temp_dir, f"{idx:06d}.npz")
                if os.path.exists(frame_path):
                    frame_data = np.load(frame_path)
                    break
        
        if frame_data is None:
            continue
        
        input_img = frame_data['input']
        recon_img = frame_data['recon']
        
        # Pad the shorter image to match the taller one's height
        if input_img.shape[0] != recon_img.shape[0]:
            max_height = max(input_img.shape[0], recon_img.shape[0])
            if input_img.shape[0] < max_height:
                # Pad input image
                pad_height = max_height - input_img.shape[0]
                input_img = np.pad(input_img, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
            elif recon_img.shape[0] < max_height:
                # Pad reconstructed image
                pad_height = max_height - recon_img.shape[0]
                recon_img = np.pad(recon_img, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        
        # Concatenate side by side
        side_by_side = np.concatenate([input_img, recon_img], axis=1)
        frames.append(side_by_side)
    
    if len(frames) == 0:
        print("Warning: No frames found to create video.")
        return
    
    # Write video using imageio
    try:
        # Try using imageio-ffmpeg if available (better quality)
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
    except Exception as e:
        # Fallback to default codec
        print(f"Warning: Could not use libx264 codec ({e}), trying default codec...")
        try:
            imageio.mimsave(output_path, frames, fps=fps)
        except Exception as e2:
            print(f"Error creating video: {e2}")
            raise


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # Check if data_path is a parquet file or a directory
    data_path = Path(args.data_path)
    if data_path.suffix.lower() == ".parquet":
        if rank == 0:
            print(f"Loading images from parquet file: {args.data_path}")
        dataset = ParquetDataset(args.data_path, pixel_key=args.pixel_key, transform=transform)
    else:
        if rank == 0:
            print(f"Loading images from ImageFolder directory: {args.data_path}")
        dataset = IndexedImageFolder(args.data_path, transform=transform)
    
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::world_size]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    sample_folder_dir = os.path.join(args.sample_dir, "-".join(folder_components))
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving video to {sample_folder_dir}")
    dist.barrier()
    
    # Temporary directory for collecting frames (will be cleaned up after video creation)
    temp_frames_dir = os.path.join(sample_folder_dir, f"temp_frames_rank_{rank}")
    if rank == 0:
        # Create temp directory for rank 0 (other ranks will create theirs when needed)
        os.makedirs(temp_frames_dir, exist_ok=True)

    loader = DataLoader(
        subset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    local_total = len(rank_indices)
    iterator = tqdm(loader, desc="Stage1 recon", total=math.ceil(local_total / args.per_proc_batch_size)) if rank == 0 else loader

    # Collect frames in memory (index -> (input_img, recon_img))
    frames_dict = {}
    
    with torch.inference_mode():
        for images, indices in iterator:
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True) # [B, C, H, W]
            
            # Get input images [B, H, W, C]
            input_np = images.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() 
            
            with autocast(**autocast_kwargs):
                latents = rae.encode(images) # [B, D, H, W], [4, 768, 16, 16]
                recon = rae.decode(latents)  # [B, C, H, W]
                import pdb; pdb.set_trace()
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            for input_img, recon_img, idx in zip(input_np, recon_np, indices_list):
                # Store frames in memory
                frames_dict[idx] = (input_img, recon_img)
    
    # Save frames temporarily to disk (needed for distributed gathering)
    os.makedirs(temp_frames_dir, exist_ok=True)
    for idx, (input_img, recon_img) in frames_dict.items():
        np.savez(os.path.join(temp_frames_dir, f"{idx:06d}.npz"), input=input_img, recon=recon_img)

    dist.barrier()
    if rank == 0:
        # Collect all frames from all ranks and create video
        print("Collecting frames from all ranks and creating video...")
        video_path = os.path.join(sample_folder_dir, "input_vs_reconstructed.mp4")
        create_video_from_temp_frames(sample_folder_dir, video_path, requested, fps=args.video_fps)
        print(f"Video saved to {video_path}")
        
        # Clean up temporary frame directories
        for r in range(world_size):
            temp_dir = os.path.join(sample_folder_dir, f"temp_frames_rank_{r}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images, or a parquet file.")
    parser.add_argument("--pixel-key", type=str, default="observations.images.front_img_1", help="Key in parquet file containing image bytes (only used when data-path is a parquet file).")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store reconstructed samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4, help="Number of images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to reconstruct (defaults to full dataset).")
    parser.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision mode.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    parser.add_argument("--video-fps", type=int, default=30, help="Frames per second for the output video.")
    args = parser.parse_args()
    main(args)
