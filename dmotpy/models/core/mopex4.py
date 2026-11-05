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
# MOPEX4 retains the alpha/is_time seasonal cosine parameterization.
# Its sequential water path and PET allocation are implementation-specific;
# see the audit comparison with the MARRMoT ODE class.
# ================================================================

MOPEX4_PARAMS_BOUNDS = {
    "tcrit":   [-3.0, 3.0],    # Snowfall & snowmelt temperature threshold [°C]
    "ddf":     [0.0,  20.0],   # Degree-day factor [mm/°C/d]
    "s2max":   [1.0,  2000.0], # Maximum soil moisture storage [mm]      → Sb1
    "tw":      [0.0,  1.0],    # Groundwater leakage rate [d⁻¹]
    "alpha":   [0.0,  1.0],    # Intercepted fraction of rainfall [-]    → i_alpha
    "is_time": [1.0,  365.0],  # Timing of maximum interception [d]      → i_s
    "tu":      [0.0,  1.0],    # Slow flow routing rate [d⁻¹]
    "se":      [0.05, 0.95],   # Root zone ET capacity as fraction of s3max [-]
    "s3max":   [1.0,  2000.0], # Root zone (subsurface) storage capacity → Sb2
    "tc":      [0.0,  1.0],    # Mean residence rate [d⁻¹]
}

MOPEX4_PARAMS_DESC = {
    "tcrit":   "Temperature threshold for snow/rain partitioning and melt [°C]",
    "ddf":     "Degree-day factor [mm/°C/d]",
    "s2max":   "Maximum soil moisture storage [mm]                       (= Sb1)",
    "tw":      "Groundwater leakage rate [d⁻¹], flux = tw * S_soil",
    "alpha":   "Mean interception fraction [-], seasonal cosine modulation",
    "is_time": "Day-of-year of maximum interception [d]                  (= i_s)",
    "tu":      "Slow flow routing rate [d⁻¹], flux = tu * S_sub",
    "se":      "Root zone ET capacity fraction [-], ET2 Smax = se * s3max",
    "s3max":   "Root zone (subsurface) storage capacity [mm]             (= Sb2)",
    "tc":      "Mean residence rate [d⁻¹], flux = tc * S",
}

MOPEX4_LIU_INTERCEPTION_NAMES = ("S_eff", "c")
MOPEX4_LEGACY_INTERCEPTION_NAMES = ("alpha", "is_time")


def validate_mopex4_parameter_schema(
    parameter_names: tuple[str, ...] | list[str], *, legacy_f0: bool = False
) -> None:
    """Reject accidental reuse of the wrong MOPEX4 interception schema.

    A raw vector has the same length in both formulations (10 parameters), so
    callers that persist parameter names must persist and validate the schema
    explicitly.  ``legacy_f0=True`` requests the original ``(alpha, is_time)``
    schema (the current restored MOPEX4); ``legacy_f0=False`` requests the
    Liu ``(S_eff, c)`` schema (retained for loading pre-restore checkpoints).
    """
    names = tuple(parameter_names)
    expected_slots = MOPEX4_LEGACY_INTERCEPTION_NAMES if legacy_f0 else MOPEX4_LIU_INTERCEPTION_NAMES
    actual_slots = (names[4], names[5]) if len(names) > 5 else ()
    if actual_slots != expected_slots:
        mode = "original alpha/is_time" if legacy_f0 else "Liu S_eff/c"
        raise ValueError(
            f"MOPEX4 parameter schema mismatch: expected {mode} slots "
            f"{expected_slots}, got {actual_slots}. Explicit schema routing is required."
        )


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize snow, soil, subsurface, fast-routing, and slow-routing stores.

    They correspond to MARRMoT stores S1-S5; Python names the snow store Sn,
    the soil store S1, the subsurface store S2, and the routing stores Sc1/Sc2.
    """
    return (
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sn
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S1 (soil)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # S2 (subsurface)
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc1
        torch.zeros((n_grid, nmul), device=device) + nearzero,  # Sc2
    )


# ================================================================
# 2. Interception Flux Function — original seasonal formulation
# ================================================================

def interception_4(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Seasonal interception potential using the alpha/is_time cosine form.

    MARRMoT uses ``max(0, alpha + (1-alpha)*cos(2*pi*(t*dt-is_time)/tmax))``
    times liquid rainfall. This implementation applies a scaled softplus lower
    bound, then limits interception to the rainfall amount. With alpha=0, the
    raw fraction is the seasonal cosine and its positive part produces
    interception during part of the year; alpha=1 gives a constant fraction of
    one. The softplus is a smooth approximation, not an exact replacement for
    the hard maximum.

    ``doy`` is the supplied day-of-year channel; ``is_time`` is the peak phase
    in days and ``tmax`` is the cycle length in days.
    """
    del nearzero  # preserved for call compatibility; unused in the original kernel
    rad          = 2.0 * torch.pi * (doy - is_time) / tmax
    interc_frac  = alpha + (1.0 - alpha) * torch.cos(rad)

    # Scaled softplus approximates the hard lower bound but is nonzero at zero.
    interc_frac_pos = F.softplus(interc_frac * 50.0) / 50.0   # Smooth lower bound.

    # Bound interception by the available liquid rainfall.
    flux_i = torch.minimum(interc_frac_pos * flux_pr, flux_pr)
    return flux_i


# ================================================================
# 3. Main Model Step Function
# ================================================================

def mopex4_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters — order matches MOPEX4_PARAMS_BOUNDS keys
    tcrit: torch.Tensor,     # tcrit
    ddf: torch.Tensor,       # ddf
    Sb1: torch.Tensor,       # s2max: soil-store capacity.
    tw: torch.Tensor,        # tw
    alpha: torch.Tensor,     # alpha → i_alpha (mean interception fraction)
    is_time: torch.Tensor,   # is_time → i_s (interception peak day-of-year)
    tu: torch.Tensor,        # tu
    Se: torch.Tensor,        # se
    Sb2: torch.Tensor,       # s3max: subsurface-store capacity.
    tc: torch.Tensor,        # tc
    # States
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    delta_t: float = 1.0,
    nearzero: float = 1e-6,
    *,
    doy: torch.Tensor = None,
    phase_cos: torch.Tensor = None,
    phase_sin: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-day explicit MOPEX-4 step with seasonal interception.

    Interception uses the alpha/is_time seasonal cosine form followed by a
    scaled-softplus lower bound. The day-of-year input is passed directly to
    the kernel; MARRMoT uses elapsed time ``t*dt`` and a hard maximum.

        frac   = alpha + (1 - alpha) * cos(2π*(doy - is_time)/tmax)
        I_pot  = softplus(50*frac)/50 * Pr        (smooth lower clamp)
        I      = min(I_pot, Pr, PET)

    It is calculated from post-partition liquid rainfall. This implementation
    allocates interception first and limits it by PET; ET1 and ET2 share the
    remaining PET demand. Net rain plus melt then enters the soil store. This
    order and PET budget differ from the MARRMoT ODE class, which evaluates
    interception, ET, and runoff from common ODE states. ``phase_cos`` and
    ``phase_sin`` are accepted but unused.


        I             = min(I_pot, PET)
        PET_after_I   = PET - I
        ET1           = evap_7(S1, Sb1, PET_after_I)
        PET_after_ET1 = PET_after_I - ET1
        ET2           = evap_7(S2, se*s3max, PET_after_ET1)



        I       = min(I_pot, PET)
        Pr_net  = Pr - I
        soil    = Pr_net + qn        (net rainfall + snowmelt enters S1)
        q1f     = saturation_1(soil, S1, Sb1)   (event input is net, not gross)

    Flux order is interception, soil input, ET1, saturation runoff, recharge;
    the subsurface sequence is recharge, overflow, baseflow, and ET2.
    """
    del phase_cos, phase_sin  # unused in the restored forward

    # Nonnegative guards for incoming states.
    Sn  = F.relu(Sn)
    S1  = F.relu(S1)
    S2  = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)

    # ============================================================
    # Snow Bucket (Sn = MATLAB S1)
    # MATLAB: dS1 = ps - qn
    # ============================================================

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    # Rainfall and snowfall use complementary fractions of P.

    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    # Melt is limited by the snow store before snowfall is added.

    Sn      = Sn + flux_ps
    Sn_new  = Sn - flux_qn

    # ============================================================
    # Soil Bucket (S1 = MATLAB S2)
    # MATLAB: dS2 = pr + qn - et1 - i - q1f - qw
    # Allocate interception from Pr, add net rainfall plus melt, then apply
    # ET1, saturation runoff, and recharge in sequence.
    # ============================================================

    # Step 1: interception-first water path.
    # I is allocated from the exact hard budget min(I_pot, PET) and removed
    # from the liquid rainfall BEFORE it enters the soil bucket:
    #     Pr -> I (evaporative loss) -> Pr_net = Pr - I -> soil input.
    i_pot = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(i_pot, PET)               # exact hard budget limiter
    flux_pr_net = flux_pr - flux_i
    pet_after_i = PET - flux_i

    soil_input = flux_pr_net + flux_qn
    S1 = S1 + soil_input

    # Step 2: ET1 consumes the remaining shared PET demand after I.
    flux_et1 = evap_7(S1, Sb1, pet_after_i, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1
    pet_remaining_after_et1 = pet_after_i - flux_et1

    # Step 3: Saturation runoff uses net rainfall plus snowmelt as its input.
    flux_q1f = saturation_1(soil_input, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S1)
    S1 = S1 - flux_q1f

    # Step 4: Recharge; recharge_3 returns tw times soil storage.
    flux_qw = recharge_3(tw, S1)          # Parameter bounds constrain tw to [0, 1].
    S1_new  = S1 - flux_qw

    # ============================================================
    # Subsurface Bucket (S2 = MATLAB S3)
    # MATLAB: dS3 = qw - et2 - q2f - q2u
    # Subsurface-store order: recharge, saturation runoff, baseflow, then ET2.
    # ============================================================

    S2 = S2 + flux_qw

    # Step 1: Subsurface saturation runoff from the post-input store.
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f

    # Step 2: Subsurface outflow; baseflow_1 returns tu times storage.
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u

    # Step 3: ET2 uses PET remaining after interception and ET1.
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, pet_remaining_after_et1, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new   = S2 - flux_et2

    # ============================================================
    # Routing Buckets
    # The fast store receives q1f and q2f; the slow store receives q2u.
    # ============================================================

    Sc1      = Sc1 + flux_q1f + flux_q2f
    flux_qf  = baseflow_1(tc, Sc1)
    Sc1_new  = Sc1 - flux_qf

    Sc2      = Sc2 + flux_q2u
    flux_qs  = baseflow_1(tc, Sc2)
    Sc2_new  = Sc2 - flux_qs

    # ============================================================
    # Output
    # Total ET includes interception as well as ET1 and ET2.
    # ============================================================
    Q_total  = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
