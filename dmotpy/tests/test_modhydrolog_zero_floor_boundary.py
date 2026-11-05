"""Regression for MODHYDROLOG post-update zero floors and preserved initialization."""

import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY


NEARZERO = 1.0e-6


def test_initial_nearzero_is_preserved_but_process_floors_allow_zero() -> None:
    entry = CORE_MODEL_REGISTRY["modhydrolog"]
    dtype = torch.float64
    device = torch.device("cpu")
    initialized = entry.init_fn(1, 1, device, NEARZERO)
    assert all(torch.equal(state, torch.full_like(state, NEARZERO)) for state in initialized)

    parameters = [
        torch.full((1, 1), sum(bounds) / 2.0, dtype=dtype, device=device)
        for bounds in entry.param_bounds.values()
    ]
    zero = torch.zeros((1, 1), dtype=dtype, device=device)
    initial = tuple(state.to(dtype=dtype) for state in initialized)
    output = entry.step_fn(zero, zero, zero, *parameters, *initial)
    next_states = output[2:]

    assert all(torch.isfinite(state).all() for state in next_states)
    assert all((state >= 0.0).all() for state in next_states)
    assert any(torch.equal(state, torch.zeros_like(state)) for state in next_states[1:])
