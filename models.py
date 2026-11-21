"""
Deep Convolutional Generative Adversarial Network (DCGAN) Architecture
Implemented following the foundational architecture by Radford et al. (2015).
"""

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """Custom weight initialization according to DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    DCGAN Generator Network:
    Maps 100-dimensional latent noise vector z into a 3x64x64 synthetic RGB face image.
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.main = nn.Sequential(
            # Input is latent vector Z: (nz) x 1 x 1 -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # Normalizes output to [-1, 1]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)
        return self.main(input)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network:
    Classifies an input 3x64x64 RGB face image as real (1.0) or fake/synthetic (0.0).
    """

    def __init__(self, nc: int = 3, ndf: int = 64) -> None:
        super().__init__()
        self.nc = nc
        self.ndf = ndf

        self.main = nn.Sequential(
            # Input: (nc) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),  # Probability [0, 1]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input).view(-1, 1).squeeze(1)
