import argparse
import math
import os
import sys
import shutil
from typing import List
from io import BytesIO
from pathlib import Path
import imageio
from termcolor import cprint
from einops import rearrange


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

from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


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

    # Load RAE
    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    # Load data
    data_path = '/coc/flash7/bli678/Projects/egowm/results/dinowm/eval_outputs/2025-12-11/20-12-53/rollout_latents/000.npz'
    data = np.load(data_path)
    latents = data['visual']   # (num_frames, num_patches, emb_dim) [20, 256, 768]
    z = torch.from_numpy(latents).float().to(device) 
    
    gt_imgs = data['gt_imgs']  # (num_frames, C, H, W) [20, 3, 256, 256]
    gt_imgs = torch.from_numpy(gt_imgs).float().to(device)
    gt_imgs = gt_imgs.add(1).div(2).clamp_(0, 1)

    # RAE decoder reconstruction
    z_recon_frames = []
    gt_recon_frames = []
    input_frames = []
    for t in range(z.size(0)):
        # reconstruct latents
        z_t = z[t].unsqueeze(0)  # [1, P, D]
        b, n, c = z_t.shape
        h = w = int(math.sqrt(n))
        z_input = z_t.transpose(1, 2).view(b, c, h, w)  # [1, D, H, W] [1, 384, 16, 16]

        with autocast(**autocast_kwargs):
            z_recon = rae.decode(z_input) # [1, C, H, W] [1, 3, 256, 256]
        z_recon = z_recon.clamp(0, 1) 
        z_recon_np = z_recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # [1, H, W, C] [1, 256, 256, 3]
        z_recon_frames.append(z_recon_np[0])  # [H, W, C]
        
        # reconstruct images
        gt_img_t = gt_imgs[t].unsqueeze(0)  # [1, C, H, W]
        with autocast(**autocast_kwargs):
            gt_latent = rae.encode(gt_img_t) # [1, D, H, W] [1, 384, 16, 16]
            gt_recon = rae.decode(gt_latent)  # [1, C, H, W] [1, 3, 256, 256]
        gt_recon = gt_recon.clamp(0, 1) 
        gt_recon_np = gt_recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # [1, H, W, C] [1, 256, 256, 3]
        gt_recon_frames.append(gt_recon_np[0])  # [H, W, C]
        
        # input frames
        img_input = gt_img_t.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy() # [1, H, W, C]
        input_frames.append(img_input[0])  # [H, W, C]
    
    z_recon_frames = np.stack(z_recon_frames, axis=0)       # [T, H, W, C]
    gt_recon_frames = np.stack(gt_recon_frames, axis=0)     # [T, H, W, C]
    input_frames = np.stack(input_frames, axis=0)           # [T, H, W, C]
    vis_recons = np.concatenate([input_frames, gt_recon_frames, z_recon_frames], axis=2)  # [T, H, W*3, C]

    # Save reconstructed video
    sample_dir = Path(args.sample_dir)
    imageio.mimwrite(sample_dir / "reconstructed_video.mp4", vis_recons, fps=args.video_fps, quality=8)
    print(f"Reconstructed video saved to {sample_dir / 'reconstructed_video.mp4'}")
    
    cprint("======= Finish =======", "green", attrs=["bold"])
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    # parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images, or a parquet file.")
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