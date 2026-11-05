"""Statistical metrics and bootstrap procedures for manuscript analyses."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

from .paths import KGE_EPS, N_BOOT_BASIN, N_BOOT_BLOCK, SEED_BOOT


def kge_from_sufficient(s: np.ndarray, eps: float = KGE_EPS) -> np.ndarray:
    """Exact reconstruction of streaming_kge (eps=0.1) from pooled sufficient statistics.
    
    Input s is [..., 6] holding (n, sum_sim, sum_obs, sum_sim2, sum_obs2, sum_simobs).
    """
    s = np.asarray(s, dtype=np.float64)
    n, sp, so = s[..., 0], s[..., 1], s[..., 2]
    sp2, so2, sc = s[..., 3], s[..., 4], s[..., 5]

    sn = np.maximum(n, 1.0)
    centered_p = np.maximum(sp2 - sp * sp / sn, 0.0)
    centered_o = np.maximum(so2 - so * so / sn, 0.0)
    denom_n = np.maximum(n - 1.0, 1.0)

    std_p = np.sqrt(np.maximum(centered_p / denom_n, 1.0e-24))
    std_o = np.sqrt(np.maximum(centered_o / denom_n, 1.0e-24))
    cov_scale = np.sqrt(np.maximum(centered_p * centered_o, 1.0e-24))

    r = (sc - sp * so / sn) / (cov_scale + eps)
    beta = (sp / sn) / (so / sn + eps)
    gamma = std_p / (std_o + eps)
    distance_sq = (r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2
    return 1.0 - np.sqrt(np.maximum(distance_sq, 1.0e-24))


def bootstrap_G_ratio(rec: np.ndarray, base: np.ndarray, vbs: np.ndarray,
                      n_boot: int = N_BOOT_BASIN, seed: int = SEED_BOOT) -> Tuple[float, float, float]:
    """Frozen G_set ratio with basin bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    r = np.asarray(rec, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    v = np.asarray(vbs, dtype=np.float64)
    num = r - b
    den = v - b
    n = len(r)
    d_mean = den.mean()
    point = float(num.mean() / d_mean) if d_mean != 0 else np.nan
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        dm = den[idx].mean()
        draws[i] = num[idx].mean() / dm if dm != 0 else np.nan
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOT_BASIN, seed: int = SEED_BOOT) -> Tuple[float, float]:
    """Percentile 95% CI for the mean of paired differences resampling basins."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=np.float64)
    n = len(d)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = d[idx].mean()
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    return float(lo), float(hi)
