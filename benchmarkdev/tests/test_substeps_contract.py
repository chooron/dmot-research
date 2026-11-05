import sys
from pathlib import Path

import torch

DEV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEV / "src"))

from benchmarkdev.core import MODELS, PARAM_INFO
from benchmarkdev.substeps import DAY_VALUE_ROLES, StorageSubcycling, parameter_time_units


def _linear_reservoir_step(
    P, T, PET, insc, coeff, sq, smsc, sub, crak, k, S1, S2, nearzero=1e-6
):
    del T, PET, insc, coeff, sq, smsc, sub, crak, nearzero
    q = k * S1
    return q, torch.zeros_like(q), S1 + P - q, S2


class _LinearReservoir:
    model_name = "simhyd"
    step_fn = staticmethod(_linear_reservoir_step)


def test_all_registry_parameters_have_reviewed_scaling_rows():
    assert len(DAY_VALUE_ROLES) == 17
    assert len(MODELS) == 36
    assert sum(len(PARAM_INFO[name]) for name in MODELS) == 271
    for name in MODELS:
        factors, audit = parameter_time_units(name)
        assert set(factors) == set(PARAM_INFO[name])
        assert len(audit) == len(PARAM_INFO[name])


def test_known_unit_and_semantic_exceptions():
    for name, parameter in (
        ("hbv96", "cfmax"),
        ("mopex2", "ddf"),
        ("mopex3", "ddf"),
        ("mopex4", "ddf"),
        ("mopex5", "ddf"),
    ):
        assert parameter_time_units(name)[0][parameter] == -1
    assert parameter_time_units("susannah1")[0]["a"] == 0
    for name, parameter in (
        ("mopex4", "is_time"),
        ("mopex5", "is_time"),
        ("vic", "ishift"),
        ("hbv96", "maxbas"),
        ("ihacres", "tau_q"),
    ):
        assert parameter_time_units(name)[0][parameter] == 0


def test_substeps_converge_to_linear_reservoir_analytic_solution():
    dtype = torch.float64
    P = torch.tensor([2.0], dtype=dtype)
    T = PET = torch.tensor([0.0], dtype=dtype)
    params = [torch.tensor([0.0], dtype=dtype) for _ in range(6)]
    k = torch.tensor([0.3], dtype=dtype)
    S1 = torch.tensor([10.0], dtype=dtype)
    S2 = torch.tensor([1.0], dtype=dtype)
    nearzero = 1e-12
    args = [P, T, PET, *params, k, S1, S2, nearzero]
    exact = S1 * torch.exp(-k) + (P / k) * (1 - torch.exp(-k))
    errors = {}

    for N in (1, 2, 4, 8, 24, 96, 384):
        model = _LinearReservoir()
        wrapper = StorageSubcycling(model, N)
        try:
            with torch.no_grad():
                result = model.step_fn(*args)
            assert torch.isfinite(result[2]).all()
            errors[N] = float((result[2] - exact).abs().max())
        finally:
            wrapper.restore()

    assert errors[1] > errors[2] > errors[4] > errors[8] > errors[24] > errors[96] > errors[384]
    assert errors[384] < errors[1] / 100


def test_gsfb_smooth_cap_width_scales_with_substep_duration():
    import inspect
    from benchmarkdev.core import STATE_INFO

    seen_tau = []

    def gsfb_step(*args, **kwargs):
        n_parameters = len(PARAM_INFO["gsfb"])
        n_states = STATE_INFO["gsfb"]
        states = args[3+n_parameters:3+n_parameters+n_states]
        seen_tau.append(kwargs.get("tau", 1e-3))
        zero = torch.zeros_like(states[0])
        return (zero, zero, *states)

    signature = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                 for name in ("P", "T", "PET", *PARAM_INFO["gsfb"],
                              *(f"S{i+1}" for i in range(STATE_INFO["gsfb"]))) ]
    signature += [inspect.Parameter("nearzero", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1e-6),
                  inspect.Parameter("tau", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1e-3)]
    gsfb_step.__signature__ = inspect.Signature(signature)

    class _SmoothModel:
        model_name = "gsfb"
        step_fn = staticmethod(gsfb_step)

    args = [torch.tensor([1.0], dtype=torch.float64) for _ in range(3+len(PARAM_INFO["gsfb"]))]
    args += [torch.tensor([1.0], dtype=torch.float64) for _ in range(STATE_INFO["gsfb"])]
    args += [1e-6]
    model = _SmoothModel()
    wrapper = StorageSubcycling(model, 4)
    try:
        with torch.no_grad():
            model.step_fn(*args)
    finally:
        wrapper.restore()
    assert seen_tau == [0.001 / 4] * 4



def test_tcm_daily_mean_precipitation_is_scaled_per_substep():
    import inspect
    from benchmarkdev.core import STATE_INFO

    seen_mean_p = []

    def tcm_step(*args, **kwargs):
        n_parameters = len(PARAM_INFO["tcm"])
        n_states = STATE_INFO["tcm"]
        states = args[3+n_parameters:3+n_parameters+n_states]
        mean_p = kwargs["mean_P"]
        seen_mean_p.append(float(mean_p[0]))
        zero = torch.zeros_like(states[0])
        return (mean_p, zero, *states)

    signature = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                 for name in ("P", "T", "PET", *PARAM_INFO["tcm"],
                              *(f"S{i+1}" for i in range(STATE_INFO["tcm"]))) ]
    signature += [inspect.Parameter("nearzero", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1e-6),
                  inspect.Parameter("mean_P", inspect.Parameter.KEYWORD_ONLY)]
    tcm_step.__signature__ = inspect.Signature(signature)

    class _TCMModel:
        model_name = "tcm"
        step_fn = staticmethod(tcm_step)

    args = [torch.tensor([1.0], dtype=torch.float64) for _ in range(3+len(PARAM_INFO["tcm"]))]
    args += [torch.tensor([1.0], dtype=torch.float64) for _ in range(STATE_INFO["tcm"])]
    args += [1e-6]
    model = _TCMModel()
    wrapper = StorageSubcycling(model, 4)
    try:
        with torch.no_grad():
            output = model.step_fn(*args, mean_P=torch.tensor([100.0], dtype=torch.float64))
    finally:
        wrapper.restore()
    assert seen_mean_p == [25.0] * 4
    assert float(output[0][0]) == 100.0
