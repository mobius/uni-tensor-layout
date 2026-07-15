"""Host-side feature matrix cleaning for dataprep pipelines (E3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CleanStats:
    nan_count: int
    clipped_count: int
    shape: tuple[int, int]
    col_mean_before: np.ndarray
    col_std_before: np.ndarray
    col_mean_after: np.ndarray
    col_std_after: np.ndarray


def make_dirty_matrix(
    m: int,
    k: int,
    *,
    rng: np.random.Generator,
    nan_frac: float = 0.02,
    outlier_frac: float = 0.01,
    outlier_scale: float = 25.0,
) -> np.ndarray:
    """Synthetic dirty features: Gaussian + NaNs + large outliers."""
    x = rng.standard_normal((m, k))
    n = m * k
    n_nan = int(n * nan_frac)
    n_out = int(n * outlier_frac)
    if n_nan > 0:
        flat = rng.choice(n, size=n_nan, replace=False)
        x.ravel()[flat] = np.nan
    if n_out > 0:
        # only finite positions for outliers
        finite_idx = np.flatnonzero(np.isfinite(x.ravel()))
        if finite_idx.size:
            pick = rng.choice(finite_idx, size=min(n_out, finite_idx.size), replace=False)
            x.ravel()[pick] *= outlier_scale
    return x


def clean_features(
    x: np.ndarray,
    *,
    clip_sigma: float = 6.0,
    fill: str = "median",
) -> tuple[np.ndarray, CleanStats]:
    """Fill NaN by column median/mean, then clip |z| > clip_sigma (robust)."""
    x = np.array(x, dtype=np.float64, copy=True)
    m, k = x.shape
    nan_count = int(np.isnan(x).sum())
    mean_b = np.nanmean(x, axis=0)
    std_b = np.nanstd(x, axis=0)
    std_b = np.where(std_b < 1e-12, 1.0, std_b)

    for j in range(k):
        col = x[:, j]
        mask = np.isnan(col)
        if mask.any():
            if fill == "mean":
                fill_v = float(np.nanmean(col))
            else:
                fill_v = float(np.nanmedian(col))
            if not np.isfinite(fill_v):
                fill_v = 0.0
            col[mask] = fill_v
            x[:, j] = col

    # clip relative to column median / MAD-ish: use mean/std after fill
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    clipped = np.abs(z) > clip_sigma
    clipped_count = int(clipped.sum())
    if clipped_count:
        x = np.where(clipped, mean + np.sign(z) * clip_sigma * std, x)

    stats = CleanStats(
        nan_count=nan_count,
        clipped_count=clipped_count,
        shape=(m, k),
        col_mean_before=mean_b,
        col_std_before=std_b,
        col_mean_after=x.mean(axis=0),
        col_std_after=x.std(axis=0),
    )
    return x, stats


def random_projection_matrix(
    k: int,
    n_out: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tall/wide random matrix with orthonormal columns (or rows) via QR."""
    if n_out <= k:
        g = rng.standard_normal((k, n_out))
        q, _ = np.linalg.qr(g, mode="reduced")
        return np.ascontiguousarray(q[:, :n_out], dtype=np.float64)
    # wide: orthonormal rows
    g = rng.standard_normal((n_out, k))
    q, _ = np.linalg.qr(g.T, mode="reduced")
    return np.ascontiguousarray(q, dtype=np.float64)
