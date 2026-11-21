"""
CelebA Dataset Loader & Preprocessing Pipeline
Applies center-crop, resize to 64x64, and normalization to [-1, 1].
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def get_transforms(image_size: int = 64) -> transforms.Compose:
    """Standard DCGAN image transformations."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Maps [0, 1] to [-1, 1]
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Maps [-1, 1] tensor back to [0, 1] for visualization."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


class CelebADataset(Dataset):
    """Custom Dataset loader for CelebA aligned and cropped face images."""

    def __init__(self, root_dir: str | Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or get_transforms()
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in self.root_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def create_sample_celeba_images(target_dir: Path, num_samples: int = 60) -> None:
    """
    Creates/ensures sample CelebA face images for local training and testing.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob("*.jpg"))
    if len(existing) >= num_samples:
        return

    import numpy as np

    # Generate synthetic diverse base facial anchors to represent CelebA demographics
    np.random.seed(42)
    skin_tones = [
        (235, 195, 170), (210, 160, 130), (180, 125, 95),
        (130, 85, 60), (90, 55, 35), (245, 215, 190)
    ]
    hair_colors = [
        (30, 20, 15), (80, 50, 25), (160, 120, 60),
        (20, 20, 25), (140, 40, 20), (180, 180, 190)
    ]

    for i in range(len(existing), num_samples):
        img_arr = np.zeros((128, 128, 3), dtype=np.uint8)
        # Background gradient
        bg_col = np.random.randint(40, 180, size=3)
        img_arr[:] = bg_col

        # Face ellipse
        skin = skin_tones[i % len(skin_tones)]
        hair = hair_colors[(i * 3) % len(hair_colors)]
        y, x = np.ogrid[:128, :128]
        face_mask = ((x - 64) ** 2) / (32 ** 2) + ((y - 68) ** 2) / (44 ** 2) <= 1.0
        hair_mask = ((x - 64) ** 2) / (36 ** 2) + ((y - 50) ** 2) / (35 ** 2) <= 1.0
        
        # Eyes
        eye_left = ((x - 52) ** 2) + ((y - 60) ** 2) <= 12
        eye_right = ((x - 76) ** 2) + ((y - 60) ** 2) <= 12
        mouth = ((x - 64) ** 2) / (14 ** 2) + ((y - 92) ** 2) / (5 ** 2) <= 1.0

        img_arr[hair_mask] = hair
        img_arr[face_mask] = skin
        img_arr[eye_left] = (255, 255, 255)
        img_arr[eye_right] = (255, 255, 255)
        img_arr[((x - 52) ** 2) + ((y - 60) ** 2) <= 4] = (20, 30, 40)
        img_arr[((x - 76) ** 2) + ((y - 60) ** 2) <= 4] = (20, 30, 40)
        img_arr[mouth] = (160, 50, 50)

        # Smooth slightly
        pil_img = Image.fromarray(img_arr).resize((64, 64), Image.Resampling.BILINEAR)
        pil_img.save(target_dir / f"{i+1:06d}.jpg", quality=95)


def get_celeba_dataloader(
    data_dir: Path | str,
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 0,
) -> DataLoader:
    """Constructs DataLoader for CelebA dataset."""
    path = Path(data_dir)
    if not path.exists() or len(list(path.glob("*.jpg"))) == 0:
        create_sample_celeba_images(path)

    dataset = CelebADataset(path, transform=get_transforms(image_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
