import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Shared hydrologic flux functions used by this model.
from ..flux.saturation import saturation_14, saturation_2
from ..flux.interflow import interflow_5
from ..flux.baseflow import baseflow_1
from ..flux.split import split_1, split_2
from ..flux.evap import evap_21

# Parameter ranges based on MARRMoT m_28_xinanjiang_12p_4s.
XINANJIANG_PARAMS_BOUNDS = {
    "aim": [0.0, 1.0],
    "par_a": [-0.49, 0.49],
    "par_b": [0.0, 10.0],
    "stot": [1.0, 2000.0],
    "fwm": [0.01, 0.99],
    "flm": [0.01, 0.99],
    "par_c": [0.01, 0.99],
    "ex": [0.0, 10.0],
    "ki": [0.0, 1.0],
    "kg": [0.0, 1.0],
    "ci": [0.0, 1.0],
    "cg": [0.0, 1.0],
}

# Parameter descriptions.
XINANJIANG_PARAMS_DESC = {
    "aim": "Fraction impervious area [-]",
    "par_a": "Tension water distribution inflection parameter [-]",
    "par_b": "Tension water distribution shape parameter [-]",
    "stot": "Total soil moisture storage (W+S) [mm]",
    "fwm": "Fraction of Stot that is Wmax [-]",
    "flm": "Fraction of wmax that is LM [-]",
    "par_c": "Fraction of LM for second evaporation change [-]",
    "ex": "Free water distribution shape parameter [-]",
    "ki": "Free water interflow parameter [d-1]",
    "kg": "Free water groundwater parameter [d-1]",
    "ci": "Interflow time coefficient [d-1]",
    "cg": "Baseflow time coefficient [d-1]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the four initial storage states for Xinanjiang."""
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4


def xinanjiang_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters.
    aim: torch.Tensor,
    par_a: torch.Tensor,
    par_b: torch.Tensor,
    stot: torch.Tensor,
    fwm: torch.Tensor,
    flm: torch.Tensor,
    par_c: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    # States.
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """One sequential daily update for the Xinanjiang model.

    Outgoing fluxes are limited to available storage where explicitly coded.
    """
    # Derived storage capacities and threshold.
    wmax = fwm * stot
    smax = (1.0 - fwm) * stot
    lm = flm * wmax

    # --------------------------------------------------------------------------
    # 1. Precipitation partitioning
    # --------------------------------------------------------------------------
    flux_rb = split_1(aim, P, nearzero)
    flux_pi = split_2(aim, P, nearzero)  # P - flux_rb

    # --------------------------------------------------------------------------
    # 2. Tension-water store S1 and runoff generation
    # --------------------------------------------------------------------------
    flux_r = saturation_14(par_a, par_b, S1, wmax, flux_pi, nearzero)
    flux_r = torch.minimum(flux_r, flux_pi)  # Limit runoff to the incoming flux.

    # Evaporation is evaluated from the current S1 storage.
    # Net input to S1 after runoff.
    s1_net_in = flux_pi - flux_r
    
    # Water temporarily available in S1 for the evaporation calculation.
    s1_avail = S1 + s1_net_in
    s1_avail = F.relu(s1_avail)  # Prevent negative available storage.
    
    flux_e_pot = evap_21(lm, par_c, S1, PET, nearzero)
    flux_e = torch.minimum(flux_e_pot, s1_avail)  # Limit evaporation to available water.

    # Update S1.
    S1_new = s1_avail - flux_e
    S1_new = torch.clamp(S1_new, min=0.0)

    # --------------------------------------------------------------------------
    # 3. Free-water store S2, updated sequentially
    # --------------------------------------------------------------------------
    # Evaluate saturation excess from the beginning-of-step S2 storage.
    flux_rs = saturation_2(S2, smax, ex, flux_r, nearzero)
    S2_curr = F.relu(S2 + flux_r - flux_rs)

    # Interflow is evaluated from S2 after saturation excess.
    flux_ri = saturation_2(S2_curr, smax, ex, S2_curr * ki, nearzero)
    flux_ri = torch.minimum(flux_ri, S2_curr)
    S2_curr = S2_curr - flux_ri

    # Groundwater recharge is evaluated after the interflow update.
    flux_rg = saturation_2(S2_curr, smax, ex, S2_curr * kg, nearzero)
    flux_rg = torch.minimum(flux_rg, S2_curr)

    S2_new = torch.clamp(S2_curr - flux_rg, min=0.0)

    # --------------------------------------------------------------------------
    # 4. Routing stores S3 and S4
    # --------------------------------------------------------------------------
    # S3 routes interflow.
    S3_tmp = S3 + flux_ri
    flux_qi = interflow_5(ci, S3_tmp, nearzero)
    flux_qi = torch.minimum(flux_qi, S3_tmp)  # Limit outflow to available storage.
    S3_new = S3_tmp - flux_qi
    S3_new = torch.clamp(S3_new, min=0.0)

    # S4 routes groundwater recharge.
    S4_tmp = S4 + flux_rg
    flux_qg = baseflow_1(cg, S4_tmp, nearzero)
    flux_qg = torch.minimum(flux_qg, S4_tmp)  # Limit outflow to available storage.
    S4_new = S4_tmp - flux_qg
    S4_new = torch.clamp(S4_new, min=0.0)

    # --------------------------------------------------------------------------
    # 5. Output aggregation
    # --------------------------------------------------------------------------
    Qsim = flux_rb + flux_rs + flux_qi + flux_qg
    Ea = flux_e

    return Qsim, Ea, S1_new, S2_new, S3_new, S4_new