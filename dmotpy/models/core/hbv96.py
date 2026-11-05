import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.melt import melt_1
from ..flux.refreeze import refreeze_1
from ..flux.infiltration import infiltration_3
from ..flux.excess import excess_1
from ..flux.capillary import capillary_1
from ..flux.evap import evap_3
from ..flux.recharge import recharge_2
from ..flux.interflow import interflow_2
from ..flux.percolation import percolation_1
from ..flux.baseflow import baseflow_1

# Parameter range dictionary (matching MARRMoT m_37_hbv_15p_5s)
HBV96_PARAMS_BOUNDS = {
    "tt": [-3.0, 5.0],           # TT, threshold temperature for snowfall [oC]
    "tti": [0.0, 17.0],          # TTI, interval length of rain-snow spectrum [oC]
    "ttm": [-3.0, 3.0],          # TTM, threshold temperature for snowmelt [oC]
    "cfr": [0.0, 1.0],           # CFR, coefficient of refreezing of melted snow [-]
    "cfmax": [0.0, 20.0],        # CFMAX, degree-day factor of snowmelt and refreezing [mm/oC/d]
    "whc": [0.0, 1.0],           # WHC, maximum water holding content of snow pack [-]
    "cflux": [0.0, 4.0],         # CFLUX, maximum rate of capillary rise [mm/d]
    "fc": [1.0, 2000.0],         # FC, maximum soil moisture storage [mm]
    "lp": [0.05, 0.95],          # LP, wilting point as fraction of FC [-]
    "beta": [0.0, 10.0],         # BETA, non-linearity coefficient of upper zone recharge [-]
    "k0": [0.0, 1.0],            # K0, runoff coefficient from upper zone [d-1]
    "alpha": [0.0, 4.0],         # ALPHA, non-linearity coefficient of runoff from upper zone [-]
    "perc": [0.0, 20.0],         # PERC, maximum rate of percolation to lower zone [mm/d]
    "k1": [0.0, 1.0],            # K1, runoff coefficient from lower zone [d-1]
    "maxbas": [1.0, 120.0],      # MAXBAS, flow routing delay [d]
}

# Parameter physical descriptions
HBV96_PARAMS_DESC = {
    "tt": "Threshold temperature for snowfall [oC]",
    "tti": "Interval length of rain-snow spectrum [oC]",
    "ttm": "Threshold temperature for snowmelt [oC]",
    "cfr": "Coefficient of refreezing of melted snow [-]",
    "cfmax": "Degree-day factor of snowmelt and refreezing [mm/oC/d]",
    "whc": "Maximum water holding content of snow pack [-]",
    "cflux": "Maximum rate of capillary rise [mm/d]",
    "fc": "Maximum soil moisture storage [mm]",
    "lp": "Wilting point as fraction of FC [-]",
    "beta": "Non-linearity coefficient of upper zone recharge [-]",
    "k0": "Runoff coefficient from upper zone [d-1]",
    "alpha": "Non-linearity coefficient of runoff from upper zone [-]",
    "perc": "Maximum rate of percolation to lower zone [mm/d]",
    "k1": "Runoff coefficient from lower zone [d-1]",
    "maxbas": "Flow routing delay [d]",
}

def create_initial_state(
    n_grid: int, 
    nmul: int, 
    device: torch.device, 
    nearzero: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial states for HBV-96 model.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


def rain_snow_partition(
    P: torch.Tensor,
    T: torch.Tensor,
    tt: torch.Tensor,
    tti: torch.Tensor,
    nearzero: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complementary rain/snow fluxes across the HBV96 TTI interval.

    Scheme A: the original HBV-96 linear transition
    ``clamp(0.5 + (tt - T) / tti, 0, 1)`` applies for every ``tti > 0``; the
    zero-width hard temperature boundary (snow at ``T <= tt``) is selected
    only when ``tti == 0`` exactly.  ``nearzero`` is retained for signature
    compatibility and no longer selects the hard branch.  Computing one
    fraction and deriving rain by subtraction guarantees partition closure.
    """
    zero_width = tti == 0
    safe_width = torch.where(zero_width, torch.ones_like(tti), tti)
    snow_fraction = torch.clamp(0.5 + (tt - T) / safe_width, min=0.0, max=1.0)
    hard_snow_fraction = (T <= tt).to(dtype=P.dtype)
    snow_fraction = torch.where(zero_width, hard_snow_fraction, snow_fraction)
    snow = P * snow_fraction
    rain = P - snow
    return rain, snow

def hbv96_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameters
    tt: torch.Tensor,
    tti: torch.Tensor,
    ttm: torch.Tensor,
    cfr: torch.Tensor,
    cfmax: torch.Tensor,
    whc: torch.Tensor,
    cflux: torch.Tensor,
    fc: torch.Tensor,
    lp: torch.Tensor,
    beta: torch.Tensor,
    k0: torch.Tensor,
    alpha: torch.Tensor,
    perc: torch.Tensor,
    k1: torch.Tensor,
    maxbas: torch.Tensor,
    # State variables
    S1: torch.Tensor,
    S2: torch.Tensor,
    S3: torch.Tensor,
    S4: torch.Tensor,
    S5: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # 1) Snow routine (S1 & S2)
    flux_rf, flux_sf = rain_snow_partition(P, T, tt, tti, nearzero=nearzero)
    flux_melt = melt_1(cfmax, ttm, T, S1)
    flux_refr = refreeze_1(cfr, cfmax, ttm, T, S2)
    
    # S1 Update
    S1 = S1 + flux_sf + flux_refr - flux_melt
    S1 = torch.clamp(S1, min=0.0)
    
    # S2 Update
    S2_potential = S2 + flux_rf + flux_melt - flux_refr
    capacity = whc * S1
    flux_se = torch.relu(S2_potential - capacity)
    S2 = torch.minimum(S2_potential, capacity)
    S2 = torch.clamp(S2, min=0.0)

    # 2) Interaction (S3 <-> S4)
    flux_cf_pot = capillary_1(cflux, S3, fc, S4)
    flux_cf = torch.minimum(flux_cf_pot, S4)
    
    S4 = S4 - flux_cf
    S3 = S3 + flux_cf

    # 3) Soil routine (S3)
    S3 = S3 + flux_se
    
    # Evaporation
    flux_ea_pot = evap_3(lp, S3, fc, PET)
    flux_ea = torch.minimum(flux_ea_pot, S3) # Limit ET to available soil water.
    S3 = S3 - flux_ea # Subtract the flux.
    
    # Recharge
    flux_r_pot = recharge_2(beta, S3, fc, flux_se) 
    
    # Limit soil recharge to the water remaining in S3.
    flux_r = torch.minimum(flux_r_pot, S3)
    S3 = S3 - flux_r # Subtract the flux.
    
    # Keep the state at or above the numerical floor after this outflow.
    S3 = torch.clamp(S3, min=0.0)

    # 4) Upper zone (S4)
    S4 = S4 + flux_r
    
    # Percolation
    flux_perc_pot = percolation_1(perc, S4)
    flux_perc = torch.minimum(flux_perc_pot, S4)
    S4 = S4 - flux_perc # Subtract the flux.
    
    # Interflow (Q0)
    flux_q0_pot = interflow_2(k0, S4, alpha)
    flux_q0 = torch.minimum(flux_q0_pot, S4)
    S4 = S4 - flux_q0 # Subtract the flux.
    
    # Keep the state at or above the numerical floor after this outflow.
    S4 = torch.clamp(S4, min=0.0)

    # 5) Lower zone (S5)
    S5 = S5 + flux_perc
    
    # Baseflow (Q1)
    flux_q1_pot = baseflow_1(k1, S5)
    flux_q1 = torch.minimum(flux_q1_pot, S5)
    S5 = S5 - flux_q1
    
    S5 = torch.clamp(S5, min=0.0)

    # 6) Aggregation
    Qsim = flux_q0 + flux_q1
    Ea = flux_ea
    
    return Qsim, Ea, S1, S2, S3, S4, S5

