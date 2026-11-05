"""Two-stage basin weighting for the canonical per-model dPL trainer.

This module deliberately contains only the basin-level state and reducer.  It
never runs a hydrological forward pass and it never sees validation/test data.
The caller must construct :class:`BasinSpecialization` with the optimization
basin IDs and pass only those IDs to :meth:`observe`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import numpy as np


@dataclass(frozen=True)
class SpecializationConfig:
    """Validated configuration for equal, rank-sigmoid, or null weighting.

    ``warmup`` and ``ramp`` are measured in optimizer epochs for the canonical
    runner.  ``freeze`` is an absolute epoch at which target weights become
    immutable; ``None`` leaves them refreshable until training ends.
    ``ess_floor`` is an ESS ratio in ``(0, 1]``.
    """

    enabled: bool = False
    method: str = "equal"
    warmup: int = 0
    ramp: int = 0
    freeze: int | None = None
    ema_beta: float = 0.9
    update_interval: int = 1
    rank_center: float = 0.5
    rank_temperature: float = 0.1
    min_weight: float = 0.1
    ess_floor: float = 0.5
    seed: int = 42
    min_valid_observations: int = 10
    required_basin_fraction: float = 0.8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SpecializationConfig":
        """Build and validate config from a job-config subsection."""
        values = dict(raw or {})
        known_keys = {
            "enabled",
            "method",
            "warmup",
            "ramp",
            "freeze",
            "ema_beta",
            "update_interval",
            "rank_center",
            "rank_temperature",
            "min_weight",
            "ess_floor",
            "seed",
            "min_valid_observations",
            "required_basin_fraction",
        }
        unknown = sorted(set(values).difference(known_keys))
        if unknown:
            raise ValueError(f"Unknown key(s) in specialization config: {unknown}")
        result = cls(
            enabled=bool(values.get("enabled", False)),
            method=str(values.get("method", "equal")),
            warmup=int(values.get("warmup", 0)),
            ramp=int(values.get("ramp", 0)),
            freeze=None if values.get("freeze") is None else int(values["freeze"]),
            ema_beta=float(values.get("ema_beta", 0.9)),
            update_interval=int(values.get("update_interval", 1)),
            rank_center=float(values.get("rank_center", 0.5)),
            rank_temperature=float(values.get("rank_temperature", 0.1)),
            min_weight=float(values.get("min_weight", 0.1)),
            ess_floor=float(values.get("ess_floor", 0.5)),
            seed=int(values.get("seed", 42)),
            min_valid_observations=int(values.get("min_valid_observations", 10)),
            required_basin_fraction=float(values.get("required_basin_fraction", 0.8)),
        )
        if result.method not in {"equal", "rank_sigmoid", "permuted_null"}:
            raise ValueError(
                "specialization.method must be one of equal, rank_sigmoid, permuted_null"
            )
        if result.warmup < 0 or result.ramp < 0:
            raise ValueError("specialization.warmup and specialization.ramp must be >= 0")
        if result.freeze is not None and result.freeze <= result.warmup:
            raise ValueError("specialization.freeze must be after specialization.warmup")
        if not 0.0 <= result.ema_beta < 1.0:
            raise ValueError("specialization.ema_beta must be in [0, 1)")
        if result.update_interval < 1:
            raise ValueError("specialization.update_interval must be >= 1")
        if not 0.0 <= result.rank_center <= 1.0:
            raise ValueError("specialization.rank_center must be in [0, 1]")
        if result.rank_temperature <= 0.0:
            raise ValueError("specialization.rank_temperature must be > 0")
        if not 0.0 < result.min_weight <= 1.0:
            raise ValueError("specialization.min_weight must be in (0, 1]")
        if not 0.0 < result.ess_floor <= 1.0:
            raise ValueError("specialization.ess_floor must be in (0, 1]")
        if result.min_valid_observations < 1:
            raise ValueError("specialization.min_valid_observations must be >= 1")
        if not 0.0 < result.required_basin_fraction <= 1.0:
            raise ValueError("specialization.required_basin_fraction must be in (0, 1]")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return resolved plain dictionary of all configuration parameters."""
        return {
            "enabled": self.enabled,
            "method": self.method,
            "warmup": self.warmup,
            "ramp": self.ramp,
            "freeze": self.freeze,
            "ema_beta": self.ema_beta,
            "update_interval": self.update_interval,
            "rank_center": self.rank_center,
            "rank_temperature": self.rank_temperature,
            "min_weight": self.min_weight,
            "ess_floor": self.ess_floor,
            "seed": self.seed,
            "min_valid_observations": self.min_valid_observations,
            "required_basin_fraction": self.required_basin_fraction,
        }


def effective_sample_size(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ESS, ESS_ratio)`` for a one-dimensional weight vector."""
    values = weights.detach().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot calculate ESS for an empty basin set")
    total = values.sum()
    ess = total.square() / values.square().sum().clamp_min(torch.finfo(values.dtype).tiny)
    return ess, ess / values.numel()


def _ess_safe_weights(
    weights: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull weights deterministically toward one until the ESS floor holds."""
    values = weights.detach().clone().reshape(-1)
    ess, ratio = effective_sample_size(values)
    if float(ratio.item()) >= floor:
        return values, torch.ones((), dtype=values.dtype, device=values.device), ratio

    # Candidate(alpha) = 1 + alpha * (candidate - 1).  For target weights in
    # (0, 1], alpha=0 is uniform and alpha=1 is the requested specialization.
    low = 0.0
    high = 1.0
    for _ in range(80):
        alpha = (low + high) / 2.0
        trial = 1.0 + alpha * (values - 1.0)
        _, trial_ratio = effective_sample_size(trial)
        if float(trial_ratio.item()) >= floor:
            low = alpha
        else:
            high = alpha
    alpha_tensor = torch.as_tensor(low, dtype=values.dtype, device=values.device)
    adjusted = 1.0 + alpha_tensor * (values - 1.0)
    _, adjusted_ratio = effective_sample_size(adjusted)
    return adjusted, alpha_tensor, adjusted_ratio


def weighted_basin_loss(
    per_basin_loss: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce detached basin weights and per-basin losses without re-running physics."""
    losses = per_basin_loss.reshape(-1)
    basin_weights = weights.detach().reshape(-1).to(device=losses.device, dtype=losses.dtype)
    if losses.numel() != basin_weights.numel():
        raise ValueError("per_basin_loss and weights must have the same number of basins")
    valid = torch.isfinite(losses)
    if valid_mask is not None:
        valid &= valid_mask.detach().reshape(-1).to(device=losses.device, dtype=torch.bool)
    if not bool(valid.any()):
        # Keep a differentiable scalar for the optimizer while matching the
        # legacy loss convention when every basin is invalid.
        return losses.sum() * 0.0 + 1.0
    denominator = basin_weights[valid].sum().clamp_min(torch.finfo(losses.dtype).tiny)
    return (basin_weights[valid] * losses[valid]).sum() / denominator


def loss_from_per_basin_kge(
    per_basin_kge: torch.Tensor,
    *,
    invalid_penalty: float = -1_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the native KGE vector to ``(losses, valid_mask, legacy_loss)``.

    The native benchmark reducer ignores entries carrying its invalid KGE
    penalty.  This helper mirrors that behavior and exposes the vector needed
    by the specialization reducer.
    """
    kge = per_basin_kge.reshape(-1)
    valid = torch.isfinite(kge) & (kge != invalid_penalty)
    losses = 1.0 - kge
    valid_kge = torch.where(valid, kge, torch.zeros_like(kge))
    legacy_loss = 1.0 - valid_kge.sum() / valid.sum().clamp_min(1)
    return losses, valid, legacy_loss


class BasinSpecialization:
    """Per-model state machine for Stage I/Stage II basin weighting."""

    def __init__(
        self,
        train_basin_ids: Sequence[int],
        config: SpecializationConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        ids = [int(value) for value in train_basin_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("train_basin_ids must be a non-empty unique sequence")
        self.config = config
        self.basin_ids = tuple(ids)
        self._index = {basin_id: index for index, basin_id in enumerate(ids)}
        self.device = torch.device(device)
        self.dtype = dtype
        n = len(ids)
        nan = torch.full((n,), float("nan"), dtype=dtype, device=self.device)
        self.ema_loss = nan.clone()
        self.valid_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.invalid_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.sample_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self._distinct_starts: list[set[int]] = [set() for _ in range(n)]
        self.instant_loss = nan.clone()
        self.instant_sum = torch.zeros(n, dtype=dtype, device=self.device)
        self.instant_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.rank = nan.clone()
        self.target_weights = torch.ones(n, dtype=dtype, device=self.device)
        self.actual_weights = torch.ones(n, dtype=dtype, device=self.device)
        self.last_refresh_epoch: int | None = None
        self.refresh_count = 0
        self.target_frozen = False
        self.coverage_gate_open = False
        self.eligible_basin_count = 0
        self.ess_adjustment_alpha = torch.ones((), dtype=dtype, device=self.device)
        self.ess = torch.as_tensor(float(n), dtype=dtype, device=self.device)
        self.ess_ratio = torch.ones((), dtype=dtype, device=self.device)
        self.phase = "stage_I"
        self._current_epoch: int | None = None

    @property
    def n_basins(self) -> int:
        return len(self.basin_ids)

    @property
    def ema_count(self) -> torch.Tensor:
        return self.valid_count

    def _indices(self, basin_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(basin_ids):
            values = [int(value) for value in basin_ids.detach().cpu().reshape(-1).tolist()]
        else:
            values = [int(value) for value in basin_ids]
        unknown = sorted(set(values).difference(self._index))
        if unknown:
            raise ValueError(
                "specialization received basin IDs outside the explicitly marked "
                f"optimization set: {unknown[:5]}"
            )
        return torch.as_tensor(
            [self._index[value] for value in values], dtype=torch.long, device=self.device
        )

    def start_epoch(self, epoch: int) -> None:
        """Refresh targets at the configured frequency and prepare actual weights."""
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("epoch must be >= 1")
        self._current_epoch = epoch
        self.instant_sum.zero_()
        self.instant_count.zero_()

        if not self.config.enabled or self.config.method == "equal":
            self.phase = "stage_I" if epoch <= self.config.warmup else "stage_II_equal"
            self.actual_weights.fill_(1.0)
            self.target_weights.fill_(1.0)
            self.ess = torch.as_tensor(float(self.n_basins), dtype=self.dtype, device=self.device)
            self.ess_ratio.fill_(1.0)
            self.coverage_gate_open = False
            return

        if epoch <= self.config.warmup:
            self.phase = "stage_I"
            self.actual_weights.fill_(1.0)
            self.target_weights.fill_(1.0)
            self.ess = torch.as_tensor(float(self.n_basins), dtype=self.dtype, device=self.device)
            self.ess_ratio.fill_(1.0)
            self.coverage_gate_open = False
            return

        # Check data-coverage gate: sufficient basins must have enough valid observations
        eligible_mask = (self.valid_count >= self.config.min_valid_observations) & torch.isfinite(self.ema_loss)
        eligible_count = int(eligible_mask.sum().item())
        coverage_ratio = eligible_count / self.n_basins
        gate_open = coverage_ratio >= self.config.required_basin_fraction
        self.coverage_gate_open = gate_open
        self.eligible_basin_count = eligible_count

        if not gate_open:
            self.phase = "stage_I_coverage_wait"
            self.actual_weights.fill_(1.0)
            self.target_weights.fill_(1.0)
            self.ess = torch.as_tensor(float(self.n_basins), dtype=self.dtype, device=self.device)
            self.ess_ratio.fill_(1.0)
            return

        freeze_now = self.config.freeze is not None and epoch >= self.config.freeze
        if freeze_now:
            self.target_frozen = True
        should_refresh = self.last_refresh_epoch is None or (
            not self.target_frozen
            and (epoch - (self.config.warmup + 1)) % self.config.update_interval == 0
        )
        if should_refresh and not self.target_frozen:
            self._refresh_target(epoch, eligible_mask)

        if self.target_frozen:
            self.phase = "stage_II_frozen"
        else:
            self.phase = "stage_II_ramp" if self._ramp_lambda(epoch) < 1.0 else "stage_II"
        lam = self._ramp_lambda(epoch)
        proposed = 1.0 + lam * (self.target_weights - 1.0)
        safe, alpha, ratio = _ess_safe_weights(proposed, self.config.ess_floor)
        self.actual_weights.copy_(safe)
        self.ess_adjustment_alpha.copy_(alpha)
        self.ess, self.ess_ratio = effective_sample_size(self.actual_weights)
        # ``ratio`` is retained as a sanity check for the scalar path above.
        if float(self.ess_ratio.item()) + 1.0e-12 < self.config.ess_floor:
            raise RuntimeError(f"ESS floor enforcement failed: {float(ratio.item())}")

    def _ramp_lambda(self, epoch: int) -> float:
        if epoch <= self.config.warmup:
            return 0.0
        if self.config.ramp == 0:
            return 1.0
        return min(1.0, max(0.0, (epoch - self.config.warmup) / self.config.ramp))

    def _refresh_target(self, epoch: int, eligible_mask: torch.Tensor | None = None) -> None:
        if eligible_mask is None:
            eligible_mask = (self.valid_count >= self.config.min_valid_observations) & torch.isfinite(self.ema_loss)

        self.target_weights.fill_(1.0)
        self.rank.fill_(float("nan"))

        if self.config.method == "equal":
            self.last_refresh_epoch = int(epoch)
            self.refresh_count += 1
            return

        eligible_indices = torch.where(eligible_mask)[0]
        m = eligible_indices.numel()
        if m == 0:
            self.last_refresh_epoch = int(epoch)
            self.refresh_count += 1
            return

        values = self.ema_loss[eligible_indices]
        order = torch.argsort(values, stable=True)
        if m == 1:
            sub_rank = torch.tensor([0.5], dtype=self.dtype, device=self.device)
        else:
            sub_rank = torch.arange(m, dtype=self.dtype, device=self.device) / (m - 1)
        eligible_rank = torch.empty_like(values)
        eligible_rank[order] = sub_rank
        self.rank[eligible_indices] = eligible_rank

        sigmoid = torch.sigmoid(
            (eligible_rank - self.config.rank_center) / self.config.rank_temperature
        )
        eligible_candidate = self.config.min_weight + (1.0 - self.config.min_weight) * (1.0 - sigmoid)

        if self.config.method == "permuted_null":
            generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed + self.refresh_count
            )
            permutation = torch.randperm(m, generator=generator).to(device=self.device)
            eligible_candidate = eligible_candidate.index_select(0, permutation)

        candidate = torch.ones(self.n_basins, dtype=self.dtype, device=self.device)
        candidate[eligible_indices] = eligible_candidate

        candidate, alpha, _ = _ess_safe_weights(candidate, self.config.ess_floor)
        self.target_weights.copy_(candidate.detach())
        self.ess_adjustment_alpha.copy_(alpha)
        self.last_refresh_epoch = int(epoch)
        self.refresh_count += 1

    def weights_for(self, basin_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return detached actual weights for one optimization batch."""
        indices = self._indices(basin_ids)
        return self.actual_weights.index_select(0, indices).detach()

    def observe(
        self,
        basin_ids: Sequence[int] | torch.Tensor,
        per_basin_loss: torch.Tensor,
        valid_mask: torch.Tensor,
        window_starts: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Update only optimization-basin EMA, window exposure, and epoch logging state."""
        indices = self._indices(basin_ids)
        losses = per_basin_loss.detach().reshape(-1).to(device=self.device, dtype=self.dtype)
        valid = valid_mask.detach().reshape(-1).to(device=self.device, dtype=torch.bool)
        if len(indices) != losses.numel() or losses.numel() != valid.numel():
            raise ValueError("basin IDs, losses, and valid_mask must have the same length")
        valid &= torch.isfinite(losses)

        starts_list: list[int] | None = None
        if window_starts is not None:
            if torch.is_tensor(window_starts):
                starts_list = [int(v) for v in window_starts.detach().cpu().reshape(-1).tolist()]
            else:
                starts_list = [int(v) for v in window_starts]
            if len(starts_list) != len(indices):
                raise ValueError("window_starts length must match basin IDs")

        if torch.unique(indices).numel() != indices.numel():
            for step_k, (index, loss, is_valid) in enumerate(zip(indices.tolist(), losses, valid.tolist())):
                self.sample_count[index] += 1
                if starts_list is not None:
                    self._distinct_starts[index].add(starts_list[step_k])
                if not is_valid:
                    self.invalid_count[index] += 1
                    continue
                self.valid_count[index] += 1
                if int(self.valid_count[index]) == 1 or not bool(torch.isfinite(self.ema_loss[index])):
                    updated = loss
                else:
                    updated = self.config.ema_beta * self.ema_loss[index] + (1.0 - self.config.ema_beta) * loss
                self.ema_loss[index] = updated
                self.instant_loss[index] = loss
                self.instant_sum[index] += loss
                self.instant_count[index] += 1
            return

        self.sample_count.index_add_(
            0, indices, torch.ones_like(indices, dtype=self.sample_count.dtype)
        )
        if starts_list is not None:
            for idx_val, start_val in zip(indices.tolist(), starts_list):
                self._distinct_starts[idx_val].add(start_val)

        invalid_indices = indices[~valid]
        if invalid_indices.numel() > 0:
            self.invalid_count.index_add_(
                0, invalid_indices, torch.ones_like(invalid_indices, dtype=self.invalid_count.dtype)
            )

        valid_indices = indices[valid]
        if valid_indices.numel() == 0:
            return

        valid_losses = losses[valid]
        old_count = self.valid_count[valid_indices]
        old_ema = self.ema_loss[valid_indices]
        initialized = (old_count == 0) | ~torch.isfinite(old_ema)
        updated = torch.where(
            initialized,
            valid_losses,
            self.config.ema_beta * old_ema + (1.0 - self.config.ema_beta) * valid_losses,
        )
        self.ema_loss[valid_indices] = updated
        self.valid_count.index_add_(
            0, valid_indices, torch.ones_like(valid_indices, dtype=self.valid_count.dtype)
        )
        self.instant_loss[valid_indices] = valid_losses
        self.instant_sum.index_add_(0, valid_indices, valid_losses)
        self.instant_count.index_add_(
            0, valid_indices, torch.ones_like(valid_indices, dtype=self.instant_count.dtype)
        )

    def epoch_rows(self, epoch: int) -> list[dict[str, Any]]:
        """Return a compact machine-readable per-basin telemetry table."""
        actual = self.actual_weights.detach().cpu().tolist()
        target = self.target_weights.detach().cpu().tolist()
        ema = self.ema_loss.detach().cpu().tolist()
        rank = self.rank.detach().cpu().tolist()
        counts = self.instant_count.detach().cpu().tolist()
        instant_sum = self.instant_sum.detach().cpu().tolist()
        sample_counts = self.sample_count.detach().cpu().tolist()
        valid_counts = self.valid_count.detach().cpu().tolist()
        invalid_counts = self.invalid_count.detach().cpu().tolist()
        rows: list[dict[str, Any]] = []
        for i, basin_id in enumerate(self.basin_ids):
            instant = instant_sum[i] / counts[i] if counts[i] else float("nan")
            rows.append(
                {
                    "epoch": int(epoch),
                    "basin_id": int(basin_id),
                    "sample_count": int(sample_counts[i]),
                    "valid_loss_count": int(valid_counts[i]),
                    "invalid_loss_count": int(invalid_counts[i]),
                    "distinct_windows_seen": len(self._distinct_starts[i]),
                    "instant_loss": instant,
                    "ema_loss": ema[i],
                    "ema_updates": int(valid_counts[i]),
                    "rank": rank[i],
                    "target_weight": target[i],
                    "actual_weight": actual[i],
                    "history_sufficient": bool(valid_counts[i] >= self.config.min_valid_observations),
                    "coverage_gate_open": bool(self.coverage_gate_open),
                    "ess": float(self.ess.item()),
                    "ess_ratio": float(self.ess_ratio.item()),
                    "phase": self.phase,
                    "target_frozen": bool(self.target_frozen),
                    "target_refresh_epoch": self.last_refresh_epoch,
                    "ess_adjustment_alpha": float(self.ess_adjustment_alpha.item()),
                }
            )
        return rows

    def exposure_summary(self, epoch: int) -> dict[str, Any]:
        """Summary statistics of basin exposure for Phase 4 audit."""
        samples = self.sample_count.detach().cpu().numpy().astype(float)
        valids = self.valid_count.detach().cpu().numpy().astype(float)
        invalids = self.invalid_count.detach().cpu().numpy().astype(float)
        distincts = np.asarray([len(s) for s in self._distinct_starts], dtype=float)

        def _stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            cv = std / mean if mean > 0 else 0.0
            return {
                f"{prefix}_min": float(np.min(arr)),
                f"{prefix}_q25": float(np.percentile(arr, 25)),
                f"{prefix}_median": float(np.median(arr)),
                f"{prefix}_q75": float(np.percentile(arr, 75)),
                f"{prefix}_max": float(np.max(arr)),
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_cv": float(cv),
            }

        result = {
            "epoch": int(epoch),
            "n_basins": self.n_basins,
            "eligible_basin_count": self.eligible_basin_count,
            "coverage_gate_open": bool(self.coverage_gate_open),
            "phase": self.phase,
            **_stats(samples, "sample_count"),
            **_stats(valids, "valid_count"),
            **_stats(invalids, "invalid_count"),
            **_stats(distincts, "distinct_windows"),
        }
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe state without optimizer or hydrology tensors."""
        return {
            "basin_ids": list(self.basin_ids),
            "ema_loss": self.ema_loss.detach().cpu(),
            "ema_count": self.valid_count.detach().cpu(),
            "valid_count": self.valid_count.detach().cpu(),
            "invalid_count": self.invalid_count.detach().cpu(),
            "sample_count": self.sample_count.detach().cpu(),
            "distinct_windows_seen": [len(s) for s in self._distinct_starts],
            "instant_loss": self.instant_loss.detach().cpu(),
            "rank": self.rank.detach().cpu(),
            "target_weights": self.target_weights.detach().cpu(),
            "actual_weights": self.actual_weights.detach().cpu(),
            "last_refresh_epoch": self.last_refresh_epoch,
            "refresh_count": self.refresh_count,
            "target_frozen": self.target_frozen,
            "coverage_gate_open": self.coverage_gate_open,
            "eligible_basin_count": self.eligible_basin_count,
            "phase": self.phase,
        }
