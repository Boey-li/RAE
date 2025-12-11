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
    data_path = '/coc/flash7/bli678/Projects/egowm/results/dinowm/eval_outputs/2025-12-11/15-09-37/rollout_latents/001.npz'
    data = np.load(data_path)
    latents = data['visual']  # (num_frames, num_patches, emb_dim)
    z = torch.from_numpy(latents).float().to(device)  # [T, P, D] [20, 256, 384]
    z_0 = z[0].unsqueeze(0)  # [1, P, D]
    b, n, c = z_0.shape
    h = w = int(math.sqrt(n))
    z_input = z_0.transpose(1, 2).view(b, c, h, w)  # [1, D, H, W] [1, 384, 16, 16]
    
    with autocast(**autocast_kwargs):
        z_recon = rae.decode(z_input)
    import pdb; pdb.set_trace()
    
    
    
    
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