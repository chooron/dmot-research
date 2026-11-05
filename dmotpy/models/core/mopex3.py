import torch
import torch.nn.functional as F
from typing import Tuple

from .mopex1 import (
    evap_7,
    saturation_1,
    baseflow_1,
    recharge_3,
)
from .mopex2 import (
    snowfall_1,
    rainfall_1,
    melt_1,
)

# ================================================================
# 1. Parameter Configuration
# ================================================================

MOPEX3_PARAMS_BOUNDS = {
    "tcrit": [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":   [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max": [1.0,  2000.0], # Maximum soil moisture storage [mm]
    "tw":    [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":    [0.05, 0.95],   # Root zone ET capacity as fraction of s3max [-]
    "s3max": [1.0,  2000.0], # Root zone (subsurface) storage capacity [mm]
    "tc":    [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX3_PARAMS_DESC = {
    "tcrit": "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":   "Degree-day factor [mm/°C/d]",
    "s2max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S2",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S3",
    "se":    "Root zone ET capacity fraction [-], ET2 uses se * s3max as Smax",
    "s3max": "Root zone (subsurface) storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize snow, soil, subsurface, fast-routing, and slow-routing stores.

    These correspond to MARRMoT stores S1 through S5, respectively.
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sn  (S1)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S2
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S3
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc1 (S4)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc2 (S5)
    )


# ================================================================
# 2. Main Model Step Function
# ================================================================

def mopex3_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb2: torch.Tensor,     # s2max: soil-store capacity.
    tw: torch.Tensor,      # Groundwater leakage rate [d⁻¹].
    tu: torch.Tensor,      # Subsurface outflow rate [d⁻¹].
    se: torch.Tensor,      # Fraction; ET2 capacity is se times Sb3.
    Sb3: torch.Tensor,     # s3max: subsurface-store capacity.
    tc: torch.Tensor,      # Routing-reservoir outflow rate [d⁻¹].
    # States
    Sn: torch.Tensor,      # Snow store (MARRMoT S1).
    S2: torch.Tensor,      # Soil store (MARRMoT S2).
    S3: torch.Tensor,      # Subsurface store (MARRMoT S3).
    Sc1: torch.Tensor,     # Fast-routing store (MARRMoT S4).
    Sc2: torch.Tensor,     # Slow-routing store (MARRMoT S5).
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-day explicit MOPEX-3 step with five stores.

    MARRMoT state equations are dS1=ps-qn, dS2=pr+qn-et1-q1f-qw,
    dS3=qw-et2-q2f-q2u, dS4=q1f+q2f-qf, and dS5=q2u-qs. Relative to
    MOPEX-2, this model has a separate subsurface capacity, uses se*s3max as
    the ET2 capacity, and routes subsurface saturation runoff q2f to the fast
    routing store.

    This implementation applies fluxes sequentially within the day; MARRMoT
    evaluates the fluxes as rates in an ODE system.
    """

    # ── Guards ────────────────────────────────────────────────────
    Sn  = F.relu(Sn)
    S2  = F.relu(S2)
    S3  = F.relu(S3)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # The rain and snow kernels use complementary sigmoid fractions.

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    # Melt is limited by the snow available before today's snowfall is added.

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn

    # ============================================================
    # Soil Bucket (S2)
    # MATLAB: dS2 = pr + qn - et1 - q1f - qw
    # Soil-store order: add liquid input, then apply ET, saturation runoff, and recharge.
    # ============================================================

    S2 = S2 + flux_pr + flux_qn

    # Step 1: Evaporation from the soil store.
    flux_et1 = evap_7(S2, Sb2, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S2)
    S2 = S2 - flux_et1

    # Step 2: Saturation runoff from the post-ET soil store.
    flux_q1f = saturation_1(flux_pr + flux_qn, S2, Sb2, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S2)
    S2 = S2 - flux_q1f

    # Step 3: Recharge; recharge_3 returns tw*S2.
    flux_qw = recharge_3(tw, S2)         # Parameter bounds constrain tw to [0, 1].
    S2_new  = S2 - flux_qw

    # ============================================================
    # Subsurface Bucket (S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # Subsurface-store order: add recharge, then saturation runoff, baseflow, and ET2.
    #
    # MOPEX-3 adds q2f, a saturation runoff flux from the subsurface store.
    # ============================================================

    S3 = S3 + flux_qw

    # Step 1: Subsurface saturation runoff, evaluated on the post-input store.
    flux_q2f = saturation_1(flux_qw, S3, Sb3, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S3)
    S3 = S3 - flux_q2f

    # Step 2: Subsurface outflow; baseflow_1 returns tu*S3.
    flux_q2u = baseflow_1(tu, S3)
    S3 = S3 - flux_q2u

    # Step 3: ET2 uses the fractional subsurface capacity se*Sb3.
    se_abs   = se * Sb3                  # ET2's effective maximum storage [mm].
    flux_et2 = evap_7(S3, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S3)
    S3_new   = S3 - flux_et2

    # ============================================================
    # Routing Buckets
    # The fast-routing store receives q1f and q2f; the slow store receives q2u.
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs

    # ============================================================
    # Output
    # MARRMoT groups ET1/ET2 as Ea and qf/qs as streamflow.
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2

    return Q_total, ET_total, Sn_new, S2_new, S3_new, Sc1_new, Sc2_new