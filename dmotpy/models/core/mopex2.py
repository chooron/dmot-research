import torch
import torch.nn.functional as F
from typing import Tuple

from ..flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_melt_1 as melt_1,
    mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1,
)

# ================================================================
# 1. Parameter Configuration
# Parameter ordering and physical roles are mapped to MARRMoT; formulas are audited separately.
# ================================================================

MOPEX2_PARAMS_BOUNDS = {
    "tcrit": [-3.0, 3.0],     # Snowfall & snowmelt temperature threshold [°C]
    "ddf":   [0.0,  20.0],    # Degree-day factor for snowmelt [mm/°C/d]
    "s2max": [1.0,  2000.0],  # Maximum soil moisture storage [mm]
    "tw":    [0.0,  1.0],     # Groundwater leakage rate [d⁻¹]
    "tu":    [0.0,  1.0],     # Slow flow routing rate [d⁻¹]
    "se":    [1.0,  2000.0],  # Root zone storage capacity [mm]
    "tc":    [0.0,  1.0],     # Mean residence rate [d⁻¹]
}

MOPEX2_PARAMS_DESC = {
    "tcrit": "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":   "Degree-day factor [mm/°C/d], melt = ddf * max(T - tcrit, 0)",
    "s2max": "Maximum soil moisture storage [mm]",
    "tw":    "Groundwater leakage rate [d⁻¹], flux = tw * S2",
    "tu":    "Slow flow routing rate [d⁻¹],   flux = tu * S3",
    "se":    "Root zone storage capacity [mm]",
    "tc":    "Mean residence rate [d⁻¹],       flux = tc * S",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize snow, soil, subsurface, fast-routing, and slow-routing stores.

    These correspond to MARRMoT stores S1 through S5, respectively.
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
        torch.zeros((n_grid, nmul), device=device) + nearzero,
    )


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    tcrit: torch.Tensor,
    ddf: torch.Tensor,
    Sb2: torch.Tensor,     # s2max: soil-store capacity.
    tw: torch.Tensor,      # Groundwater leakage rate [d⁻¹].
    tu: torch.Tensor,      # Slow-routing rate [d⁻¹].
    Se: torch.Tensor,      # se: subsurface ET capacity.
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
    One-day explicit MOPEX-2 step with snow, soil, subsurface, and two routing stores.

    MARRMoT state equations are dS1=ps-qn, dS2=pr+qn-et1-q1f-qw,
    dS3=qw-et2-q2u, dS4=q1f-qf, and dS5=q2u-qs. This implementation applies
    fluxes sequentially within the day and uses nonnegative input guards and
    availability limits; it is not the simultaneous ODE evaluation used by
    MARRMoT.

    Soil-store order: add rain and melt, ET1, saturation runoff, recharge.
    Subsurface order: add recharge, baseflow, then ET2.
    """

    # Guards against small negative values carried from the previous step.
    Sn  = F.relu(Sn)
    S2  = F.relu(S2)
    S3  = F.relu(S3)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    # Partition precipitation into complementary rain and snow fractions.
    # MATLAB: flux_ps = snowfall_1(P, T, tcrit)
    #         flux_pr = rainfall_1(P, T, tcrit)
    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)

    # Melt is limited by the snow available before today's snowfall is added.
    # MATLAB: flux_qn = melt_1(ddf, tcrit, T, S1, delta_t)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)

    # Add snowfall before applying the melt flux.
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    # ============================================================
    # Soil Bucket (S2)
    # MATLAB: dS2 = pr + qn - et1 - q1f - qw
    # Soil-store order: add liquid input, then apply ET, saturation runoff, and recharge.
    # ============================================================

    # Add liquid rainfall and snowmelt to the soil store.
    S2 = S2 + flux_pr + flux_qn

    # Step 1: Evaporation.
    # MATLAB: flux_et1 = evap_7(S2, s2max, Ep, dt)
    flux_et1 = evap_7(S2, Sb2, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S2)   # Limit ET1 to available storage.
    S2 = S2 - flux_et1

    # Step 2: Saturation runoff uses the post-ET soil store and the original
    # liquid-rain-plus-melt input. This sequential evaluation differs from the
    # MARRMoT flux evaluation at the common ODE state.
    flux_q1f = saturation_1(flux_pr + flux_qn, S2, Sb2, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S2)   # Limit runoff to the current store.
    S2 = S2 - flux_q1f

    # Step 3: Recharge; recharge_3 returns tw*S2.
    flux_qw = recharge_3(tw, S2)             # Parameter bounds constrain tw to [0, 1].
    S2_new  = S2 - flux_qw

    # ============================================================
    # Subsurface Bucket (S3)
    # MATLAB: dS3 = qw - et2 - q2u
    # Subsurface-store order: add recharge, then baseflow and ET2.
    # ============================================================

    S3 = S3 + flux_qw

    # Step 1: Baseflow; baseflow_1 returns tu*S3.
    flux_q2u = baseflow_1(tu, S3)
    S3 = S3 - flux_q2u

    # Step 2: Evaporation.
    # MATLAB: flux_et2 = evap_7(S3, se, Ep, dt)
    flux_et2 = evap_7(S3, Se, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S3)   # Limit ET2 to available storage.
    S3_new   = S3 - flux_et2

    # ============================================================
    # Routing Buckets (Sc1 = S4, Sc2 = S5)
    # The fast and slow routing stores use baseflow_1 with parameter tc.
    # ============================================================

    Sc1      = Sc1 + flux_q1f
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
