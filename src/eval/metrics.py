"""
Compute the challenge metrics from two sets of rendered images:
  - FID            via torchmetrics FrechetInceptionDistance
  - density/coverage (and precision/recall) via vendored prdc

Both metrics share ONE InceptionV3 (torchmetrics' own), so real and fake go
through identical preprocessing (resize->299, normalize) — no drift between
the two metrics.

Images are (N, H, W, 3) uint8 numpy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from prdc import compute_prdc

# Silence the noisy "Could not initialize NNPACK! Reason: Unsupported hardware."
# warning that fires on CPU convs on this box. (The python torch.backends.nnpack
# module has no `enabled` attr; the real switch is this C++ flag.)
try:
    torch._C._set_nnpack_enabled(False)
except Exception:
    pass


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) uint8 torch tensor."""
    t = torch.from_numpy(np.ascontiguousarray(images))
    return t.permute(0, 3, 1, 2).contiguous().to(torch.uint8)


@torch.no_grad()
def _inception_features(fid: FrechetInceptionDistance, imgs: torch.Tensor,
                        device: str, batch_size: int = 32) -> np.ndarray:
    """Run torchmetrics' Inception to get (N, 2048) features for prdc."""
    feats = []
    for i in range(0, imgs.shape[0], batch_size):
        batch = imgs[i:i + batch_size].to(device)
        f = fid.inception(batch)            # (B, 2048), preprocessing handled inside
        feats.append(f.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def compute_metrics(real_images: np.ndarray, fake_images: np.ndarray,
                    nearest_k: int = 5, device: str | None = None,
                    batch_size: int = 32) -> dict:
    """Return {fid, density, coverage, precision, recall} for two image sets."""
    if device is None:
        # GPU (ROCm/CUDA) supports float64 for FID's covariance and is far faster;
        # fall back to CPU only when no GPU is present (MPS can't do float64).
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if real_images.shape[0] != fake_images.shape[0]:
        print(f"[!] N differs (real={real_images.shape[0]}, fake={fake_images.shape[0]}); "
              "FID covariance is unstable with unequal/small N.")

    real_t = _to_tensor(real_images)
    fake_t = _to_tensor(fake_images)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.set_dtype(torch.float64)  # stable covariance

    for i in range(0, real_t.shape[0], batch_size):
        fid.update(real_t[i:i + batch_size].to(device), real=True)
    for i in range(0, fake_t.shape[0], batch_size):
        fid.update(fake_t[i:i + batch_size].to(device), real=False)
    fid_score = float(fid.compute().cpu())

    real_feats = _inception_features(fid, real_t, device, batch_size)
    fake_feats = _inception_features(fid, fake_t, device, batch_size)
    prdc = compute_prdc(real_feats, fake_feats, nearest_k=nearest_k)

    return {
        "fid": fid_score,
        "density": float(prdc["density"]),
        "coverage": float(prdc["coverage"]),
        "precision": float(prdc["precision"]),
        "recall": float(prdc["recall"]),
    }
