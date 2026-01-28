"""
Conditional Stable Diffusion for Image_Rectification.

This module exposes:
  - forward_diffuse: q(x_t | x_0) forward process
  - reverse_denoise: p(x_{t-1} | x_t, condition) reverse process

The reverse step is conditioned on an image produced by the
Guided_Rectification_Network (e.g., rectified output). A user-supplied
denoise_model (typically a UNet) must accept (x_t, t, condition) and return
epsilon (predicted noise).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import argparse


class SinusoidalPosEmb(torch.nn.Module):
    """Simple sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class UNetBlock(torch.nn.Module):
    """Conv block with GroupNorm and SiLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = torch.nn.GroupNorm(8, out_ch)
        self.act = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UNetConditioned(torch.nn.Module):
    """
    Lightweight UNet-like denoiser that conditions on an image.
    Expects inputs: (x_t, t, condition) and returns epsilon.
    """

    def __init__(self, base_channels: int = 64, time_emb_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
        )

        cond_in = 3
        x_in = 3
        self.down1 = UNetBlock(x_in + cond_in, base_channels)
        self.down2 = UNetBlock(base_channels, base_channels * 2)
        self.downsample = torch.nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)

        self.mid1 = UNetBlock(base_channels * 2, base_channels * 4)
        self.mid2 = UNetBlock(base_channels * 4, base_channels * 2)

        self.up = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = UNetBlock(base_channels * 2, base_channels)
        self.out = torch.nn.Conv2d(base_channels, 3, 1)

        self.time_to_feat = torch.nn.Linear(time_emb_dim, base_channels * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # time embedding
        t_emb = self.time_mlp(t)
        t_feat = self.time_to_feat(t_emb).view(t_emb.size(0), -1, 1, 1)

        x_in = torch.cat([x, condition], dim=1)
        d1 = self.down1(x_in)
        d2 = self.down2(d1)
        d2 = self.downsample(d2)

        m = self.mid1(d2)
        m = self.mid2(m + t_feat)

        u = self.up(m)
        u = torch.cat([u, d1], dim=1)
        u = self.up1(u)
        return self.out(u)


class ConditionalStableDiffusion(torch.nn.Module):
    def __init__(
        self,
        denoise_model: torch.nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            denoise_model: network predicting epsilon given (x_t, t, condition).
            timesteps: number of diffusion steps T.
            beta_start: starting beta for linear noise schedule.
            beta_end: ending beta for linear noise schedule.
            device: optional device to place buffers on.
        """
        super().__init__()
        self.denoise_model = denoise_model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConditionalStableDiffusion":
        denoise = UNetConditioned(base_channels=args.base_channels, time_emb_dim=args.time_emb_dim)
        return cls(
            denoise_model=denoise,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=torch.device(args.device),
        )

    @torch.no_grad()
    def forward_diffuse(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: clean input image batch (B, C, H, W), assumed in [-1, 1].
            t: diffusion step indices (B,) in [0, T-1].
            noise: optional precomputed noise tensor shaped like x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar_t = self.alpha_bar.index_select(0, t).view(-1, 1, 1, 1)
        return torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise

    @torch.no_grad()
    def reverse_denoise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One reverse step: predicts x_{t-1} given x_t and a condition image.

        Args:
            xt: noisy input at step t (B, C, H, W).
            t: step indices (B,) in [0, T-1].
            condition: conditioning image (e.g., rectified output) (B, C, H, W).
            guidance_scale: scales the conditional prediction (simple classifier-free style).

        Returns:
            x_prev: predicted sample at step t-1.
            pred_eps: predicted noise epsilon.
        """
        betas_t = self.betas.index_select(0, t).view(-1, 1, 1, 1)
        alphas_t = self.alphas.index_select(0, t).view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar.index_select(0, t).view(-1, 1, 1, 1)

        # Predict noise with conditioning; assumes model signature (xt, t, condition).
        pred_eps_cond = self.denoise_model(xt, t, condition)

        # Optional simple guidance: scale conditional prediction.
        pred_eps = pred_eps_cond * guidance_scale

        # Estimate x0 from predicted noise.
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Compute mean of p(x_{t-1} | x_t, x0_pred).
        mean = (
            torch.sqrt(self.alphas.index_select(0, t).view(-1, 1, 1, 1))
            * (xt - betas_t / torch.sqrt(1 - alpha_bar_t) * pred_eps)
        )

        # Add noise except for the final step.
        noise = torch.randn_like(xt)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
        x_prev = mean + nonzero_mask * torch.sqrt(betas_t) * noise

        return x_prev, pred_eps


def example_usage():
    """
    Minimal usage example (pseudo-code):

    >>> denoise_model = UNetConditioned()
    >>> cond_diff = ConditionalStableDiffusion(
    ...     denoise_model,
    ...     timesteps=1000,
    ...     beta_start=1e-4,
    ...     beta_end=0.02,
    ... )
    >>> x0 = rectified_images  # output from Guided_Rectification_Network
    >>> t = torch.randint(0, cond_diff.timesteps, (x0.size(0),), device=x0.device)
    >>> xt = cond_diff.forward_diffuse(x0, t)
    >>> x_prev, eps = cond_diff.reverse_denoise(xt, t, condition=x0)
    """
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional Stable Diffusion (Image_Rectification)")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps (T).")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Linear beta schedule start.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Linear beta schedule end.")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels for UNet.")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Time embedding dimension.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale during reverse denoising.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    diffusion = ConditionalStableDiffusion.from_args(args)
    diffusion.to(args.device)
    # Demo shapes
    x0 = torch.randn(1, 3, 256, 256, device=args.device)
    t = torch.randint(0, diffusion.timesteps, (1,), device=args.device)
    xt = diffusion.forward_diffuse(x0, t)
    x_prev, eps = diffusion.reverse_denoise(xt, t, condition=x0, guidance_scale=args.guidance_scale)
    print("Ran conditional diffusion step:", {"xt": xt.shape, "x_prev": x_prev.shape, "eps": eps.shape})

