"""
DCGAN Adversarial Training Pipeline on CelebA
Trains Generator and Discriminator using Adam optimizers and binary cross-entropy loss.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from dataset import denormalize, get_celeba_dataloader
from models import Discriminator, Generator, weights_init


def train_dcgan(
    data_dir: Path | str = "data/celeba/img_align_celeba",
    output_dir: Path | str = "checkpoints",
    samples_dir: Path | str = "samples",
    num_epochs: int = 5,
    batch_size: int = 64,
    image_size: int = 64,
    nz: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    device: str | None = None,
) -> dict[str, Any]:
    """Executes DCGAN adversarial training loop."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_dir)
    samples_path = Path(samples_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_path.mkdir(parents=True, exist_ok=True)

    # 1. DataLoader
    dataloader = get_celeba_dataloader(data_dir, batch_size=batch_size, image_size=image_size)

    # 2. Networks
    netG = Generator(nz=nz, ngf=64, nc=3).to(device_obj)
    netG.apply(weights_init)

    netD = Discriminator(nc=3, ndf=64).to(device_obj)
    netD.apply(weights_init)

    # 3. Loss & Optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(16, nz, 1, 1, device=device_obj)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Tracking metrics
    history: dict[str, list[float]] = {"G_losses": [], "D_losses": [], "D_x": [], "D_G_z": []}
    start_time = time.time()

    print(f"Starting DCGAN Training on {device_obj} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        epoch_g_losses = []
        epoch_d_losses = []

        for i, (real_imgs, _) in enumerate(dataloader):
            current_batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device_obj)

            real_label = torch.full((current_batch_size,), 1.0, dtype=torch.float, device=device_obj)
            fake_label = torch.full((current_batch_size,), 0.0, dtype=torch.float, device=device_obj)

            # -----------------------------------------------------------
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------------------------
            netD.zero_grad()
            # Real batch
            output_real = netD(real_imgs)
            errD_real = criterion(output_real, real_label)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Fake batch
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device_obj)
            fake_imgs = netG(noise)
            output_fake = netD(fake_imgs.detach())
            errD_fake = criterion(output_fake, fake_label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # -----------------------------------------------------------
            # (2) Update Generator: maximize log(D(G(z)))
            # -----------------------------------------------------------
            netG.zero_grad()
            output = netD(fake_imgs)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            epoch_d_losses.append(errD.item())
            epoch_g_losses.append(errG.item())

        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        history["D_losses"].append(avg_d_loss)
        history["G_losses"].append(avg_g_loss)
        history["D_x"].append(D_x)
        history["D_G_z"].append(D_G_z2)

        print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {avg_d_loss:.4f} | Loss_G: {avg_g_loss:.4f} | D(x): {D_x:.4f} | D(G(z)): {D_G_z2:.4f}")

        # Save fixed noise visual sample grid
        with torch.no_grad():
            sample_fake = netG(fixed_noise).detach().cpu()
            grid = vutils.make_grid(denormalize(sample_fake), nrow=4, padding=2)
            vutils.save_image(grid, samples_path / f"epoch_{epoch:03d}.png")

    total_time = round(time.time() - start_time, 2)
    print(f"Training completed in {total_time}s.")

    # Save checkpoints
    torch.save(netG.state_dict(), output_path / "generator.pth")
    torch.save(netD.state_dict(), output_path / "discriminator.pth")

    # Save history
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save loss plot
    plt.figure(figsize=(9, 4))
    plt.plot(history["G_losses"], label="Generator Loss", color="#2563EB", lw=2)
    plt.plot(history["D_losses"], label="Discriminator Loss", color="#DC2626", lw=2)
    plt.title("DCGAN Adversarial Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(samples_path / "training_loss_curve.png")
    plt.close()

    return {
        "epochs": num_epochs,
        "total_time": total_time,
        "final_loss_G": history["G_losses"][-1],
        "final_loss_D": history["D_losses"][-1],
        "history": history,
    }


if __name__ == "__main__":
    train_dcgan()
