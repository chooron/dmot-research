import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7, evap_3
from ..flux.saturation import saturation_1
from ..flux.interflow import interflow_8

# Parameter bounds (based on MARRMoT m_03_collie2_4p_1s).
COLLIE2_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],  # Maximum soil moisture storage [mm]
    "Sfc_frac": [0.05, 0.95],  # Field capacity as fraction of Smax [-]
    "a": [0.0, 1.0],  # Subsurface runoff coefficient [d-1]
    "M": [0.05, 0.95],  # Fraction forest cover [-]
}

# Parameter descriptions.
COLLIE2_PARAMS_DESC = {
    "Smax": "Maximum soil moisture storage [mm]",
    "Sfc_frac": "Field capacity as fraction of Smax [-]",
    "a": "Subsurface runoff coefficient [d-1]",
    "M": "Fraction forest cover [-]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """Create the initial soil-moisture state for Collie2."""
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def collie2_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameter order follows COLLIE2_PARAMS_BOUNDS.
    Smax: torch.Tensor,
    Sfc_frac: torch.Tensor,
    a: torch.Tensor,
    M: torch.Tensor,
    # State variable.
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One daily step for Collie River v2.

    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward
    approach to hydrological model development. Journal of Hydrology, 254,
    174 198. doi: 10.1016/S0022-1694(01)00496-6.
    """

    # 1. Saturation-excess runoff.
    # The shared storage-threshold gate determines the runoff fraction.
    flux_qse = saturation_1(P, S1, Smax, nearzero=nearzero)
    zeros = torch.zeros_like(flux_qse)
    flux_qse = torch.clamp(flux_qse, min=zeros, max=P)

    # 2. Update available storage before evaporation.
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 3. Evaporation.
    # Bare-soil evaporation uses evap_7 on post-runoff storage and bare-soil PET.
    # Vegetation transpiration uses evap_3 on post-runoff storage and vegetated PET.
    pet_bare = (1.0 - M) * PET
    pet_veg = M * PET

    flux_eb = evap_7(S1_tmp, Smax, pet_bare, nearzero=nearzero)
    flux_ev = evap_3(Sfc_frac, S1_tmp, Smax, pet_veg, nearzero=nearzero)

    # Limit combined evaporation by available storage and PET.
    flux_ea_total = flux_eb + flux_ev
    flux_ea_total = torch.minimum(flux_ea_total, S1_tmp - nearzero)
    flux_ea_total = torch.minimum(flux_ea_total, PET)
    flux_ea_total = F.relu(flux_ea_total)

    # 4. Interflow from the post-evaporation store.
    S1_tmp2 = S1_tmp - flux_ea_total
    S1_tmp2 = torch.clamp(S1_tmp2, min=nearzero)

    # flux_qss = interflow_8(S1_tmp2, a, Sfc_frac * Smax).
    Sfc = Sfc_frac * Smax
    flux_qss = interflow_8(S1_tmp2, a, Sfc, nearzero=nearzero)
    flux_qss = torch.minimum(flux_qss, S1_tmp2 - nearzero)
    flux_qss = F.relu(flux_qss)

    # 5. Update the final store.
    S1_new = S1_tmp2 - flux_qss
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 6. Aggregate and return outputs.
    Qsim = flux_qse + flux_qss
    Ea = flux_ea_total

    return Qsim, Ea, S1_new
