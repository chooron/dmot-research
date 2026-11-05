import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.saturation import saturation_7, saturation_1
from ..flux.evap import evap_3
from ..flux.interflow import interflow_10
from ..flux.baseflow import baseflow_4

# Parameter range dictionary (based on MARRMoT m_14_topmodel_7p_2s)
TOPMODEL_PARAMS_BOUNDS = {
    "suzmax": [1.0, 2000.0],  # Max soil moisture storage in unsatured zone [mm]
    "st": [0.05, 0.95],  # Threshold fraction [-]
    "kd": [0.0, 1.0],  # Leakage coefficient [mm/d]
    "q0": [0.1, 200.0],  # Zero deficit base flow speed [mm/d]
    "f": [0.0, 1.0],  # Baseflow scaling coefficient [mm-1]
    "chi": [1.0, 7.5],  # Gamma distribution parameter [-]
    "phi": [0.1, 5.0],  # Gamma distribution parameter [-]
}

# Parameter description dictionary
TOPMODEL_PARAMS_DESC = {
    "suzmax": "Maximum soil moisture storage in unsaturated zone [mm]",
    "st": "Threshold for flow generation and evap change as fraction of suzmax [-]",
    "kd": "Leakage to saturated zone flow coefficient [mm/d]",
    "q0": "Zero deficit base flow speed [mm/d]",
    "f": "Baseflow scaling coefficient [mm-1]",
    "chi": "Gamma distribution parameter [-]",
    "phi": "Gamma distribution parameter [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create initial states for TOPMODEL.
    S1: Unsaturated storage
    S2: Saturated zone deficit (0 = fully saturated)
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2


def topmodel_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters matching TOPMODEL_PARAMS_BOUNDS keys
    suzmax: torch.Tensor,
    st: torch.Tensor,
    kd: torch.Tensor,
    q0: torch.Tensor,
    f: torch.Tensor,
    chi: torch.Tensor,
    phi: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Daily sequential TOPMODEL step; S2 is represented as a deficit store."""

    # ==========================================================================
    # 1. Surface Runoff (S2 Control)
    # ==========================================================================
    mu_fixed = torch.tensor(3.0, device=P.device, dtype=P.dtype)
    lambda_para = chi * phi + mu_fixed
    
    # Saturation-excess overland flow.
    # Bound the runoff flux to non-negative values and the day's precipitation.
    flux_qof = saturation_7(
        chi, phi, mu_fixed, lambda_para, f, S2, P, nearzero=nearzero
    )
    flux_qof = torch.clamp(flux_qof, min=torch.zeros_like(P), max=P)

    # Effective precipitation entering the unsaturated store.
    flux_peff = P - flux_qof

    # ==========================================================================
    # 2. Unsaturated Zone (S1: Positive Storage)
    # ==========================================================================
    
    # 2.1 Evaporation
    # Potential evaporation from the unsaturated store.
    flux_ea_pot = evap_3(st, S1, suzmax, PET, nearzero=nearzero)
    
    # Water available in S1 above its nearzero floor.
    s1_free = F.relu(S1 - nearzero)
    
    # Limit actual evaporation to water available above the floor.
    flux_ea = torch.minimum(flux_ea_pot, s1_free)
    
    # 2.2 Add effective precipitation and subtract actual evaporation from S1.
    S1_tmp = S1 + flux_peff - flux_ea
    
    # 2.3 Saturation excess
    # Evaluate saturation excess from the post-inflow, post-evaporation storage.
    flux_qex = saturation_1(flux_peff, S1_tmp, suzmax, nearzero=nearzero)
    flux_qex = torch.clamp(flux_qex, min=torch.zeros_like(flux_peff), max=flux_peff)

    # Subtract the saturation-excess flux from S1.
    S1_tmp = S1_tmp - flux_qex

    # 2.4 Interflow from S1 to the deficit store S2.
    threshold_s1 = st * suzmax
    capacity_s1 = suzmax - threshold_s1

    flux_qv_pot = interflow_10(
        S1_tmp, kd, threshold_s1, capacity_s1, nearzero=nearzero
    )
    
    # Limit interflow by both the available S1 water and the receiving deficit.
    s1_free_now = F.relu(S1_tmp - nearzero)
    
    # S2 is a deficit store. Include the concurrent baseflow release when
    # calculating how much interflow it can receive during this step.
    flux_qb_pot = baseflow_4(q0, f, S2, nearzero=nearzero)
    
    # Receiving capacity is the remaining deficit plus this step's baseflow.
    s2_space = F.relu(S2 - nearzero) + flux_qb_pot
    
    # Limit qv by its potential rate, available S1 water, and S2 capacity.
    flux_qv = torch.minimum(flux_qv_pot, s1_free_now)
    flux_qv = torch.minimum(flux_qv, s2_space)

    # Update S1 after interflow.
    S1_new = S1_tmp - flux_qv
    # The preceding limiter keeps this state at or above nearzero; the clamp
    # guards against floating-point round-off at that boundary.
    S1_new = torch.clamp(S1_new, min=nearzero)

    # ==========================================================================
    # 3. Saturated Zone (S2: Deficit Store)
    # ==========================================================================
    # Baseflow increases the deficit; interflow reduces it.
    
    # Re-evaluate baseflow from the unchanged beginning-of-step deficit.
    flux_qb = baseflow_4(q0, f, S2, nearzero=nearzero)
    
    # Update the deficit after baseflow and interflow.
    S2_new = S2 + flux_qb - flux_qv
    
    # The receiving-capacity limiter keeps the updated deficit above nearzero.
    S2_new = torch.clamp(S2_new, min=nearzero)

    # ==========================================================================
    # 4. Output aggregation
    # ==========================================================================
    Qsim = flux_qof + flux_qex + flux_qb
    Ea = flux_ea

    return Qsim, Ea, S1_new, S2_new
