from __future__ import annotations

import copy

import torch
from torch import nn

import benchmarkparam.scripts.canonical_v2.run_canonical_v2_model as runner
from benchmarkpub.specialization import BasinSpecialization, SpecializationConfig


class _IdentityHydro(nn.Module):
    def forward(self, x_dict, _parameters):
        return {"streamflow": x_dict["x_phy"][..., 0]}


def _optimizer_snapshot(optimizer: torch.optim.Optimizer) -> dict:
    return copy.deepcopy(optimizer.state_dict())


def test_internal_validation_and_oob_evaluation_are_forward_only() -> None:
    network = nn.Linear(2, 1, bias=True).double()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    optimizer.zero_grad()
    network(torch.ones(4, 2, dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    before_params = [parameter.detach().clone() for parameter in network.parameters()]
    before_optimizer = _optimizer_snapshot(optimizer)

    specialization = BasinSpecialization(
        [1, 2],
        SpecializationConfig.from_mapping({"enabled": True, "method": "rank_sigmoid"}),
        device="cpu",
    )
    before_specialization = specialization.state_dict()

    attrs = torch.ones(2, 2, dtype=torch.float64)
    forcing = torch.arange(8 * 2 * 3, dtype=torch.float64).reshape(8, 2, 3) / 10.0 + 1.0
    target = forcing[..., 0].clone()
    hydro = _IdentityHydro()

    runner.evaluate_internal_validation(hydro, network, attrs, forcing, target, warmup_days=0)
    runner.CURRENT_PHASE = runner.Phase.EVAL
    runner.evaluate_test_partition("gr4j", hydro, network, attrs, forcing, target, warmup_days=0)
    runner.CURRENT_PHASE = runner.Phase.TRAIN

    for before, after in zip(before_params, network.parameters()):
        torch.testing.assert_close(before, after.detach())
    after_optimizer = _optimizer_snapshot(optimizer)
    assert after_optimizer["param_groups"] == before_optimizer["param_groups"]
    assert after_optimizer["state"].keys() == before_optimizer["state"].keys()
    for parameter_id, before_state in before_optimizer["state"].items():
        for name, before_value in before_state.items():
            after_value = after_optimizer["state"][parameter_id][name]
            if torch.is_tensor(before_value):
                torch.testing.assert_close(before_value, after_value)
            else:
                assert after_value == before_value
    after_specialization = specialization.state_dict()
    assert before_specialization["ema_count"].equal(after_specialization["ema_count"])
    assert before_specialization["target_weights"].equal(after_specialization["target_weights"])
