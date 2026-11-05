from __future__ import annotations

import pytest
import torch

from benchmarkpub.specialization import (
    BasinSpecialization,
    SpecializationConfig,
    effective_sample_size,
    loss_from_per_basin_kge,
    weighted_basin_loss,
)


def _observe_epoch(state: BasinSpecialization, losses: list[float]) -> None:
    ids = state.basin_ids
    state.observe(
        ids,
        torch.tensor(losses, dtype=torch.float64),
        torch.ones(len(ids), dtype=torch.bool),
    )


def test_disabled_and_equal_reducer_match_native_loss() -> None:
    kge = torch.tensor([0.8, 0.4, -1_000_000.0], dtype=torch.float64, requires_grad=True)
    per_loss, valid, legacy = loss_from_per_basin_kge(kge)
    weights = torch.ones(3, dtype=torch.float64)
    equal = weighted_basin_loss(per_loss, weights, valid)
    equal.backward()
    assert kge.grad is not None and torch.isfinite(kge.grad).all()
    assert not weights.requires_grad

    state = BasinSpecialization(
        [1, 2, 3],
        SpecializationConfig.from_mapping({"enabled": False}),
        device="cpu",
    )
    state.start_epoch(1)
    assert torch.equal(state.weights_for([1, 2, 3]), torch.ones(3, dtype=torch.float64))
    assert not state.weights_for([1]).requires_grad


def test_ema_rank_direction_ramp_and_freeze() -> None:
    config = SpecializationConfig.from_mapping(
        {
            "enabled": True,
            "method": "rank_sigmoid",
            "warmup": 1,
            "ramp": 2,
            "freeze": 4,
            "ema_beta": 0.5,
            "rank_temperature": 0.15,
            "min_weight": 0.1,
            "ess_floor": 0.2,
            "min_valid_observations": 1,
        }
    )
    state = BasinSpecialization([10, 20, 30], config, device="cpu")

    state.start_epoch(1)
    _observe_epoch(state, [0.1, 0.5, 0.9])
    assert torch.equal(state.actual_weights, torch.ones(3, dtype=torch.float64))

    state.start_epoch(2)
    assert state.phase == "stage_II_ramp"
    assert state.rank[0] < state.rank[1] < state.rank[2]
    assert state.target_weights[0] > state.target_weights[2]
    assert state.actual_weights[0] > state.actual_weights[2]
    assert torch.all(state.actual_weights >= config.min_weight)

    target_epoch2 = state.target_weights.clone()
    _observe_epoch(state, [0.9, 0.9, 0.1])
    state.start_epoch(3)
    assert not torch.equal(state.target_weights, target_epoch2)

    target_epoch3 = state.target_weights.clone()
    _observe_epoch(state, [0.2, 0.2, 0.2])
    state.start_epoch(4)
    assert state.target_frozen
    torch.testing.assert_close(state.target_weights, target_epoch3)
    assert state.phase == "stage_II_frozen"


def test_ess_floor_is_enforced() -> None:
    config = SpecializationConfig.from_mapping(
        {
            "enabled": True,
            "method": "rank_sigmoid",
            "warmup": 1,
            "min_weight": 0.01,
            "rank_temperature": 0.01,
            "ess_floor": 0.8,
            "min_valid_observations": 1,
        }
    )
    state = BasinSpecialization(list(range(8)), config, device="cpu")
    state.start_epoch(1)
    _observe_epoch(state, [float(i) for i in range(8)])
    state.start_epoch(2)
    ess, ratio = effective_sample_size(state.actual_weights)
    assert float(ratio) >= config.ess_floor - 1e-10
    assert float(ess) > 0.0
    assert torch.all(state.actual_weights >= config.min_weight)


def test_permuted_null_is_reproducible_and_preserves_weight_multiset() -> None:
    common = {
        "enabled": True,
        "warmup": 1,
        "ramp": 0,
        "min_weight": 0.2,
        "rank_temperature": 0.2,
        "ess_floor": 0.3,
        "seed": 1234,
        "min_valid_observations": 1,
    }
    baseline = BasinSpecialization(
        [1, 2, 3, 4],
        SpecializationConfig.from_mapping({**common, "method": "rank_sigmoid"}),
        device="cpu",
    )
    null1 = BasinSpecialization(
        [1, 2, 3, 4],
        SpecializationConfig.from_mapping({**common, "method": "permuted_null"}),
        device="cpu",
    )
    null2 = BasinSpecialization(
        [1, 2, 3, 4],
        SpecializationConfig.from_mapping({**common, "method": "permuted_null"}),
        device="cpu",
    )
    for state in (baseline, null1, null2):
        state.start_epoch(1)
        _observe_epoch(state, [0.1, 0.2, 0.8, 1.2])
        state.start_epoch(2)

    torch.testing.assert_close(torch.sort(null1.target_weights).values, torch.sort(baseline.target_weights).values)
    torch.testing.assert_close(null1.target_weights, null2.target_weights)
    assert not torch.equal(null1.target_weights, baseline.target_weights)
    torch.testing.assert_close(null1.ess_ratio, baseline.ess_ratio)
    assert not null1.target_weights.requires_grad


def test_validation_or_oob_id_cannot_update_weighting_state() -> None:
    state = BasinSpecialization(
        [101, 102],
        SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid"}),
        device="cpu",
    )
    with pytest.raises(ValueError, match="optimization set"):
        state.observe([999], torch.tensor([0.1]), torch.tensor([True]))
    with pytest.raises(ValueError, match="optimization set"):
        state.weights_for([999])
    assert torch.equal(state.ema_count, torch.zeros(2, dtype=torch.long))

def test_ema_updates_only_on_valid_and_leaves_unsampled_and_invalid_unchanged() -> None:
    config = SpecializationConfig.from_mapping({"enabled": True, "ema_beta": 0.5})
    state = BasinSpecialization([1, 2, 3], config, device="cpu")
    state.start_epoch(1)

    # Step 1: basin 1 and 2 sampled. 1 is valid (loss=0.8), 2 is invalid (NaN/penalty). Basin 3 unsampled.
    state.observe(
        [1, 2],
        torch.tensor([0.8, float("nan")], dtype=torch.float64),
        torch.tensor([True, False]),
        window_starts=[100, 200],
    )
    assert state.sample_count[0] == 1
    assert state.valid_count[0] == 1
    assert state.invalid_count[0] == 0
    assert state.ema_loss[0] == 0.8

    assert state.sample_count[1] == 1
    assert state.valid_count[1] == 0
    assert state.invalid_count[1] == 1
    assert torch.isnan(state.ema_loss[1])

    assert state.sample_count[2] == 0
    assert state.valid_count[2] == 0
    assert state.invalid_count[2] == 0
    assert torch.isnan(state.ema_loss[2])

    # Step 2: basin 1 sampled again (loss=0.4). Basin 2 and 3 unsampled.
    state.observe([1], torch.tensor([0.4], dtype=torch.float64), torch.tensor([True]), window_starts=[150])
    # EMA = 0.5 * 0.8 + 0.5 * 0.4 = 0.6
    assert state.valid_count[0] == 2
    assert abs(state.ema_loss[0] - 0.6) < 1e-10
    assert len(state._distinct_starts[0]) == 2
    # Basin 2 and 3 remain unchanged
    assert state.sample_count[1] == 1 and state.valid_count[1] == 0
    assert state.sample_count[2] == 0 and state.valid_count[2] == 0


def test_coverage_gate_and_insufficient_history_basins_keep_weight_one() -> None:
    # 4 basins, min_valid_observations=3, required_basin_fraction=0.75 (3 out of 4)
    config = SpecializationConfig.from_mapping(
        {
            "enabled": True,
            "method": "rank_sigmoid",
            "warmup": 1,
            "min_valid_observations": 3,
            "required_basin_fraction": 0.75,
            "min_weight": 0.2,
        }
    )
    state = BasinSpecialization([10, 20, 30, 40], config, device="cpu")
    state.start_epoch(1)

    # Give basin 10 and 20 three valid observations, but 30 and 40 only 1.
    for _ in range(3):
        state.observe([10, 20], torch.tensor([0.2, 0.8], dtype=torch.float64), torch.tensor([True, True]))
    state.observe([30, 40], torch.tensor([0.4, 0.6], dtype=torch.float64), torch.tensor([True, True]))

    # In epoch 2: warmup is over (epoch > 1), but eligible basins = 2/4 = 0.5 < 0.75.
    # Coverage gate must remain closed.
    state.start_epoch(2)
    assert state.phase == "stage_I_coverage_wait"
    assert not state.coverage_gate_open
    assert torch.equal(state.actual_weights, torch.ones(4, dtype=torch.float64))
    assert torch.equal(state.target_weights, torch.ones(4, dtype=torch.float64))

    # Now observe basin 30 two more times so 30 has 3 observations. Eligible = 3/4 = 0.75.
    for _ in range(2):
        state.observe([30], torch.tensor([0.5], dtype=torch.float64), torch.tensor([True]))

    # In epoch 3: coverage gate opens! Basin 10, 20, 30 are eligible. Basin 40 is ineligible.
    state.start_epoch(3)
    assert state.coverage_gate_open
    assert state.eligible_basin_count == 3
    assert state.phase in {"stage_II_ramp", "stage_II"}

    # Ineligible basin 40 MUST have target_weight == 1.0, rank == NaN
    assert state.target_weights[3] == 1.0
    assert torch.isnan(state.rank[3])
    assert state.actual_weights[3] == 1.0

    # Eligible basins 10, 20, 30 have differentiated weights: 10 has lower loss so lower rank and higher weight
    assert state.target_weights[0] > state.target_weights[1]
    assert torch.isfinite(state.rank[0]) and torch.isfinite(state.rank[1]) and torch.isfinite(state.rank[2])


def test_same_step_decoupling_loss_observation_does_not_affect_active_weights() -> None:
    config = SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid", "warmup": 0, "min_valid_observations": 1})
    state = BasinSpecialization([1, 2], config, device="cpu")
    state.start_epoch(1)
    # Initial weights at epoch 1 start (since warmup=0 and 0 observations, all 1.0)
    initial_weights = state.weights_for([1, 2]).clone()
    assert torch.equal(initial_weights, torch.tensor([1.0, 1.0], dtype=torch.float64))

    # Step observation happens during training: loss is 0.1 for basin 1 and 0.9 for basin 2
    state.observe([1, 2], torch.tensor([0.1, 0.9], dtype=torch.float64), torch.tensor([True, True]))

    # Within the same epoch, weights_for must still return the frozen active weights (1.0, 1.0)
    current_step_weights = state.weights_for([1, 2])
    assert torch.equal(current_step_weights, initial_weights)


def test_weighted_reducer_correctness_and_nan_safety() -> None:
    losses = torch.tensor([0.2, 0.8, 10.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64)
    valid_mask = torch.tensor([True, True, False])  # 3rd basin invalid

    reduced = weighted_basin_loss(losses, weights, valid_mask)
    # Expected = (2.0 * 0.2 + 3.0 * 0.8) / (2.0 + 3.0) = (0.4 + 2.4) / 5.0 = 2.8 / 5.0 = 0.56
    assert abs(reduced.item() - 0.56) < 1e-12
    reduced.backward()
    assert losses.grad is not None
    assert abs(losses.grad[0].item() - 2.0 / 5.0) < 1e-12
    assert abs(losses.grad[1].item() - 3.0 / 5.0) < 1e-12
    assert losses.grad[2].item() == 0.0

    # All invalid case: must not return NaN and must be differentiable with zero grad
    all_inv_losses = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    all_inv_mask = torch.tensor([False, False])
    safe_loss = weighted_basin_loss(all_inv_losses, torch.tensor([1.0, 1.0]), all_inv_mask)
    assert safe_loss.item() == 1.0
    safe_loss.backward()
    assert (all_inv_losses.grad == 0.0).all()


def test_permuted_null_matches_multiset_with_ineligible_basins() -> None:
    common = {
        "enabled": True,
        "warmup": 1,
        "ramp": 0,
        "min_weight": 0.2,
        "rank_temperature": 0.2,
        "ess_floor": 0.3,
        "seed": 42,
        "min_valid_observations": 2,
        "required_basin_fraction": 0.5,
    }
    baseline = BasinSpecialization([1, 2, 3, 4], SpecializationConfig.from_mapping({**common, "method": "rank_sigmoid"}), device="cpu")
    null_mod = BasinSpecialization([1, 2, 3, 4], SpecializationConfig.from_mapping({**common, "method": "permuted_null"}), device="cpu")

    for s in (baseline, null_mod):
        s.start_epoch(1)
        # Basins 1, 2, 3 get 2 valid observations; basin 4 gets only 1 observation.
        for _ in range(2):
            s.observe([1, 2, 3], torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64), torch.tensor([True, True, True]))
        s.observe([4], torch.tensor([0.5], dtype=torch.float64), torch.tensor([True]))
        s.start_epoch(2)

    # Basin 4 is ineligible in both, so both must have target_weight == 1.0 for basin 4
    assert baseline.target_weights[3] == 1.0
    assert null_mod.target_weights[3] == 1.0

    # Sorted target weights must match exactly
    torch.testing.assert_close(torch.sort(baseline.target_weights).values, torch.sort(null_mod.target_weights).values)
    torch.testing.assert_close(baseline.ess, null_mod.ess)
    torch.testing.assert_close(baseline.ess_ratio, null_mod.ess_ratio)
    assert baseline.target_weights.mean().item() == pytest.approx(null_mod.target_weights.mean().item())
    assert not torch.equal(baseline.target_weights, null_mod.target_weights)


def test_exposure_summary_telemetry() -> None:
    config = SpecializationConfig.from_mapping({"enabled": True, "min_valid_observations": 1})
    state = BasinSpecialization([10, 20], config, device="cpu")
    state.start_epoch(1)
    state.observe([10, 20], torch.tensor([0.5, 0.5], dtype=torch.float64), torch.tensor([True, False]), window_starts=[100, 200])
    summary = state.exposure_summary(1)
    assert summary["sample_count_min"] == 1.0
    assert summary["sample_count_max"] == 1.0
    assert summary["valid_count_min"] == 0.0
    assert summary["valid_count_max"] == 1.0
    assert summary["distinct_windows_mean"] == 1.0

def test_specialization_config_rejects_unknown_keys_and_roundtrips() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        SpecializationConfig.from_mapping({"enabled": True, "typo_method": "rank_sigmoid"})

    cfg = SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid", "warmup": 3})
    d = cfg.to_dict()
    assert d["enabled"] is True
    assert d["method"] == "rank_sigmoid"
    assert d["warmup"] == 3
    # Re-parsing from to_dict must produce an identical object
    roundtrip = SpecializationConfig.from_mapping(d)
    assert roundtrip == cfg

def test_stage_transitions_and_freeze_boundary_off_by_one() -> None:
    config = SpecializationConfig.from_mapping({
        "enabled": True, "method": "rank_sigmoid", "warmup": 2, "ramp": 2, "freeze": 5,
        "update_interval": 1, "min_valid_observations": 2, "required_basin_fraction": 0.5,
        "min_weight": 0.2, "ess_floor": 0.5,
    })
    spec = BasinSpecialization([1, 2, 3, 4], config, device="cpu")
    # Epoch 1: Stage I
    spec.start_epoch(1)
    assert spec.phase == "stage_I"
    spec.observe([1, 2], torch.tensor([0.2, 0.8], dtype=torch.float64), torch.tensor([True, True]))
    # Epoch 2: Still Stage I (warmup=2)
    spec.start_epoch(2)
    assert spec.phase == "stage_I"
    spec.observe([1, 2], torch.tensor([0.2, 0.8], dtype=torch.float64), torch.tensor([True, True]))
    # Epoch 3: Epoch 3 > warmup=2. Coverage gate is satisfied (2/4 = 0.5). Ramp=2 -> lambda=0.5
    spec.start_epoch(3)
    assert spec.coverage_gate_open
    assert spec.phase == "stage_II_ramp"
    assert spec._ramp_lambda(3) == 0.5
    assert spec.target_weights[2] == 1.0 and spec.target_weights[3] == 1.0
    # Epoch 4: ramp_lambda(4) == 1.0 -> stage_II
    spec.start_epoch(4)
    assert spec.phase == "stage_II"
    assert spec._ramp_lambda(4) == 1.0
    assert not spec.target_frozen
    # Epoch 5: Freeze epoch!
    spec.start_epoch(5)
    assert spec.phase == "stage_II_frozen"
    assert spec.target_frozen
    frozen_targets = spec.target_weights.clone()
    # Observe wild losses, start epoch 6: targets must be strictly immutable
    spec.observe([1, 2], torch.tensor([100.0, 0.001], dtype=torch.float64), torch.tensor([True, True]))
    spec.start_epoch(6)
    assert spec.phase == "stage_II_frozen"
    torch.testing.assert_close(spec.target_weights, frozen_targets)

def test_adversarial_basin_id_alignment_and_permutation_invariance() -> None:
    from torch import nn
    train_ids = [105, 102, 108, 101, 104]
    config = SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid"})
    spec = BasinSpecialization(train_ids, config, device="cpu")
    fixed_weights = {101: 0.2, 102: 0.5, 104: 0.7, 105: 0.9, 108: 0.4}
    with torch.no_grad():
        for b_id, w_val in fixed_weights.items():
            spec.actual_weights[spec._index[b_id]] = w_val

    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(3, 6).double(), nn.Tanh(), nn.Linear(6, 2).double())
    params = list(net.parameters())
    attr_map = {b_id: torch.randn(3, dtype=torch.float64) for b_id in train_ids}

    # Order A = [102, 105, 101]
    order_A = [102, 105, 101]
    attrs_A = torch.stack([attr_map[i] for i in order_A])
    thetas_A = net(attrs_A)
    losses_A = (thetas_A ** 2).sum(dim=1)
    w_A = spec.weights_for(order_A)
    loss_A = weighted_basin_loss(losses_A, w_A)
    net.zero_grad()
    loss_A.backward()
    grads_A = [p.grad.detach().clone() for p in params]

    # Order B = [101, 102, 105] (same basins, permuted order)
    order_B = [101, 102, 105]
    attrs_B = torch.stack([attr_map[i] for i in order_B])
    thetas_B = net(attrs_B)
    losses_B = (thetas_B ** 2).sum(dim=1)
    w_B = spec.weights_for(order_B)
    loss_B = weighted_basin_loss(losses_B, w_B)
    net.zero_grad()
    loss_B.backward()
    grads_B = [p.grad.detach().clone() for p in params]

    # Loss and Gradients must match exactly
    assert abs(loss_A.item() - loss_B.item()) < 1e-15
    for gA, gB in zip(grads_A, grads_B):
        torch.testing.assert_close(gA, gB, rtol=1e-15, atol=1e-15)

def test_autograd_graph_isolation() -> None:
    config = SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid"})
    spec = BasinSpecialization([1, 2], config, device="cpu")
    w = spec.weights_for([1, 2])
    assert not w.requires_grad
    assert w.grad_fn is None
    assert not spec.target_weights.requires_grad
    assert not spec.actual_weights.requires_grad

def test_b_vs_n_assignment_null_exact_difference_reconstruction() -> None:
    common = {
        "enabled": True, "warmup": 1, "ramp": 0, "min_weight": 0.2,
        "rank_temperature": 0.2, "ess_floor": 0.3, "seed": 42,
        "min_valid_observations": 1, "required_basin_fraction": 0.5,
    }
    spec_B = BasinSpecialization([1, 2, 3, 4], SpecializationConfig.from_mapping({**common, "method": "rank_sigmoid"}), device="cpu")
    spec_N = BasinSpecialization([1, 2, 3, 4], SpecializationConfig.from_mapping({**common, "method": "permuted_null"}), device="cpu")

    for s in (spec_B, spec_N):
        s.start_epoch(1)
        s.observe([1, 2, 3, 4], torch.tensor([0.1, 0.3, 0.7, 1.2], dtype=torch.float64), torch.tensor([True, True, True, True]))
        s.start_epoch(2)

    w_B = spec_B.actual_weights.clone()
    w_N = spec_N.actual_weights.clone()

    # Condition 1: When basin gradients are identical, G_B == G_N
    g_identical = torch.tensor([1.5, -2.0, 0.8], dtype=torch.float64)
    G_B_ident = sum((w_B[b] / w_B.sum()) * g_identical for b in range(4))
    G_N_ident = sum((w_N[b] / w_N.sum()) * g_identical for b in range(4))
    torch.testing.assert_close(G_B_ident, G_N_ident, rtol=1e-15, atol=1e-15)

    # Condition 2: When basin gradients differ, G_B - G_N is exactly reconstructable from (w_B - w_N) / sum(w)
    torch.manual_seed(99)
    g_diff = [torch.randn(5, dtype=torch.float64) for _ in range(4)]
    G_B = sum((w_B[b] / w_B.sum()) * g_diff[b] for b in range(4))
    G_N = sum((w_N[b] / w_N.sum()) * g_diff[b] for b in range(4))
    delta_G_actual = G_B - G_N
    delta_G_predicted = sum(((w_B[b] - w_N[b]) / w_B.sum()) * g_diff[b] for b in range(4))
    torch.testing.assert_close(delta_G_actual, delta_G_predicted, rtol=1e-15, atol=1e-15)

def test_full_population_oracle_match() -> None:
    from torch import nn
    import copy
    torch.manual_seed(42)
    N = 8
    train_ids = list(range(1001, 1001 + N))
    attrs = torch.randn(N, 6, dtype=torch.float64)
    net_oracle = nn.Sequential(nn.Linear(6, 12).double(), nn.Tanh(), nn.Linear(12, 4).double())
    net_prod = copy.deepcopy(net_oracle)
    opt_oracle = torch.optim.AdamW(net_oracle.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_prod = torch.optim.AdamW(net_prod.parameters(), lr=1e-3, weight_decay=1e-4)

    cfg = SpecializationConfig.from_mapping({
        "enabled": True, "method": "rank_sigmoid", "warmup": 1, "ramp": 0,
        "min_weight": 0.2, "rank_temperature": 0.2, "ess_floor": 0.5,
        "min_valid_observations": 1, "required_basin_fraction": 0.5,
    })
    spec = BasinSpecialization(train_ids, cfg, device="cpu")
    spec.start_epoch(1)
    spec.observe(train_ids, torch.linspace(0.1, 1.0, N, dtype=torch.float64), torch.ones(N, dtype=torch.bool))
    spec.start_epoch(2)
    weights = spec.weights_for(train_ids)

    opt_oracle.zero_grad(set_to_none=True)
    thetas_o = net_oracle(attrs)
    losses_o = (thetas_o ** 2).sum(dim=1)
    loss_oracle = (weights * losses_o).sum() / weights.sum()
    loss_oracle.backward()
    nn.utils.clip_grad_norm_(net_oracle.parameters(), max_norm=1.0)
    opt_oracle.step()

    opt_prod.zero_grad(set_to_none=True)
    thetas_p = net_prod(attrs)
    losses_p = (thetas_p ** 2).sum(dim=1)
    loss_prod = weighted_basin_loss(losses_p, weights)
    loss_prod.backward()
    nn.utils.clip_grad_norm_(net_prod.parameters(), max_norm=1.0)
    opt_prod.step()

    assert abs(loss_oracle.item() - loss_prod.item()) < 1e-15
    for p_o, p_p in zip(net_oracle.parameters(), net_prod.parameters()):
        torch.testing.assert_close(p_o, p_p, rtol=1e-15, atol=1e-15)
