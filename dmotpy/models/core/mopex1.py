import torch
import torch.nn.functional as F
from typing import Tuple

from ..flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
)

# ================================================================
# 1. Parameter Configuration
# Parameter ordering and physical roles are mapped to MARRMoT; formulas are audited separately.
# ================================================================

MOPEX1_PARAMS_BOUNDS = {
    "s1max": [1.0, 2000.0],   # Maximum soil moisture storage [mm]
    "tw":    [0.0, 1.0],      # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0, 1.0],      # Slow flow routing rate [d⁻¹]
    "se":    [1.0, 2000.0],   # Root zone storage capacity [mm]
    "tc":    [0.0, 1.0],      # Mean residence rate [d⁻¹]
}

MOPEX1_PARAMS_DESC = {
    "s1max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S1",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S2",
    "se":    "Root zone storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize state variables (S1, S2, Sc1, Sc2)."""
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex1_step(
    P: torch.Tensor,
    T: torch.Tensor,       # Retained for the shared model interface; unused by MOPEX-1.
    PET: torch.Tensor,
    Sb1: torch.Tensor,     # s1max: maximum soil-store capacity.
    tw: torch.Tensor,      # Groundwater leakage rate [d⁻¹].
    tu: torch.Tensor,      # Slow-routing rate [d⁻¹].
    Se: torch.Tensor,      # se: contributing capacity for the second store.
    tc: torch.Tensor,      # Routing-reservoir outflow rate [d⁻¹].
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-day explicit MOPEX-1 step.

    MARRMoT state equations are dS1=P-et1-q1f-qw, dS2=qw-et2-q2u,
    dS3=q1f-qf, and dS4=q2u-qs. This implementation evaluates and applies
    fluxes sequentially within the day, so it is not the simultaneous ODE
    evaluation used by MARRMoT. Nonnegative input guards and flux limits are
    applied explicitly.

    Soil-store order: saturation runoff, ET1, then recharge. Subsurface order:
    baseflow, then ET2. The three routing stores receive their branch inflows
    and discharge within the same daily step.
    """

    # Guards against small negative values carried from the previous step.
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Bucket 1 (surface soil, S1); fluxes are applied sequentially.
    # ============================================================

    # Step 1: Calculate saturation runoff from pre-input S1, then add net rain.
    # The shared MOPEX kernel uses a sigmoid threshold with its own shape parameters.
    flux_q1f = saturation_1(P, S1, Sb1, nearzero=nearzero)
    S1 = S1 + P - flux_q1f          # q1f is bounded by P.

    # Step 2: Evaporation from S1 after precipitation and saturation runoff.
    # MATLAB: flux_et1 = evap_7(S1, s1max, Ep, dt)
    flux_et1 = evap_7(S1, Sb1, PET, delta_t, nearzero)
    # Retain an explicit availability limit after the flux calculation.
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1

    # Step 3: Recharge from the post-ET1 soil store; recharge_3 returns tw*S1.
    flux_qw = recharge_3(tw, S1)    # The parameter bounds constrain tw to [0, 1].
    S1_new  = S1 - flux_qw

    # ============================================================
    # Bucket 2 (Subsurface / S2)
    # ============================================================

    S2 = S2 + flux_qw               # Receive recharge.

    # Step 1: Baseflow from the subsurface store before ET2.
    flux_q2u = baseflow_1(tu, S2)   # baseflow_1 returns tu*S2.
    S2 = S2 - flux_q2u

    # Step 2: ET2 uses the post-baseflow store and remaining step state.
    # MARRMoT evaluates ET2 and q2u from the same ODE state; this ordering is
    # a sequential daily scheme, not an algebraic equivalence claim.
    flux_et2 = evap_7(S2, Se, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)   # Limit ET2 to available storage.
    S2_new   = S2 - flux_et2

    # ============================================================
    # Routing stores (Sc1=S3, Sc2=S4); each uses the linear baseflow function.
    # ============================================================

    Sc1     = Sc1 + flux_q1f
    flux_qf = baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - flux_qf

    Sc2     = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Output
    # MARRMoT groups ET1/ET2 as Ea and qf/qs as streamflow.
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new
