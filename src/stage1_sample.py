#!/usr/bin/env python3
"""
Run a stage-1 RAE reconstruction from a config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from io import BytesIO
import numpy as np
import daft
from matplotlib import pyplot as plt

import torch
from PIL import Image
from torchvision import transforms

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE

DEFAULT_IMAGE = Path("assets/pixabay_cat.png")




def get_device(explicit: str | None):
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def decode_img(bytes):
    with Image.open(BytesIO(bytes)) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def load_parquet(image_path: Path, idx: int = 0):
    df = daft.read_parquet(str(image_path))
    df = df.sort("index")
    
    pixel_key = "observations.images.front_img_1"
    img_size = 64
    img_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    
    dataset = df.to_torch_map_dataset()
    img_dict = dataset[idx][pixel_key]
    img_array = decode_img(img_dict['bytes'])     # (H, W, C) uint8
    img = Image.fromarray(img_array)
    img = img_transform(img)
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # (1, C, H, W)
    return tensor


def reconstruct(rae: RAE, image: torch.Tensor):
    with torch.no_grad():
        latent = rae.encode(image)
        recon = rae.decode(latent)
    return latent, recon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    # load image
    if args.image.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = load_image(args.image).to(device)
    elif args.image.suffix.lower() == ".parquet":
        image = load_parquet(args.image).to(device)

    # RAE reconstruction
    latent, recon = reconstruct(rae, image)

    recon = recon.clamp(0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image pair
    # Input: (1, C, H, W) -> (H, W, C)
    input_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
    input_np = np.clip(input_np, 0.0, 1.0)
    
    # Reconstructed: (1, C, H, W) -> (H, W, C)
    recon_np = recon.cpu().squeeze(0).permute(1, 2, 0).numpy()
    recon_np = np.clip(recon_np, 0.0, 1.0)
    
    # Create figure with side-by-side images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(recon_np)
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved input and reconstruction to {args.output.resolve()}")
    print(f"Input shape: {tuple(image.shape)}, latent shape: {tuple(latent.shape)}, recon shape: {tuple(recon.shape)}")


if __name__ == "__main__":
    main()
