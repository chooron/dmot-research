import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.evap import evap_7
from ..flux.saturation import saturation_1

# Parameter bounds (based on MARRMoT m_01_collie1_1p_1s).
COLLIE1_PARAMS_BOUNDS = {
    "Smax": [1.0, 2000.0],
}

# Parameter descriptions.
COLLIE1_PARAMS_DESC = {
    "Smax": "Maximum soil moisture storage [mm]",
}


def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[torch.Tensor]:
    """Create the initial soil-moisture state for Collie1."""
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return (S1,)


def collie1_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    # Parameter order follows COLLIE1_PARAMS_BOUNDS.
    Smax: torch.Tensor,
    # State variable.
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One daily step for Collie River 1 (the traditional bucket model).

    Model reference:
    Jothityangkoon, C., M. Sivapalan, and D. Farmer (2001), Process controls
    of water balance variability in a large semi-arid catchment: downward 
    approach to hydrological model development. Journal of Hydrology, 254,
    174-198. doi: 10.1016/S0022-1694(01)497-6.
    """

    # 1. Saturation-excess runoff.
    # The shared threshold gate determines the fraction of P that runs off.
    flux_qse = saturation_1(P, S1, Smax, nearzero=nearzero)
    flux_qse = torch.clamp(flux_qse, min=0.0)

    # 2. Update available storage before evaporation.
    # S1_tmp = S1 + P - flux_qse.
    S1_tmp = S1 + P - flux_qse
    S1_tmp = torch.clamp(S1_tmp, min=nearzero)

    # 3. Evaporation from the updated store.
    # evap_7 uses the updated storage, a capped relative-storage ratio, and PET.
    flux_ea = evap_7(S1_tmp, Smax, PET, nearzero=nearzero)
    
    # Limit evaporation by available storage and PET.
    flux_ea = torch.minimum(flux_ea, S1_tmp - nearzero)
    flux_ea = torch.minimum(flux_ea, PET)
    flux_ea = F.relu(flux_ea)

    # 4. Update the final store.
    S1_new = S1_tmp - flux_ea
    S1_new = torch.clamp(S1_new, min=nearzero)

    # 5. Aggregate and return outputs.
    Qsim = flux_qse
    Ea = flux_ea

    return Qsim, Ea, S1_new