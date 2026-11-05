import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Shared flux functions used by the MODHYDROLOG step.
from ..flux.evap import evap_1, evap_2
from ..flux.interception import interception_modhydrolog as interception_1
from ..flux.infiltration import infiltration_1, infiltration_2
from ..flux.interflow import interflow_1
from ..flux.recharge import recharge_1
from ..flux.saturation import saturation_1

from ..flux.depression import depression_1
from ..flux.exchange import exchange_1, exchange_3
from ..flux.baseflow import baseflow_1

# Parameter bounds based on MARRMoT m_36_modhydrolog_15p_5s.
MODHYDROLOG_PARAMS_BOUNDS = {
    "insc": [0.1, 5.0],  # Maximum interception capacity [mm]
    "coeff": [0.0, 600.0],  # Maximum infiltration loss parameter [mm]
    "sq": [0.0, 15.0],  # Infiltration loss exponent [-]
    "smsc": [1.0, 2000.0],  # Maximum soil moisture capacity [mm]
    "sub": [0.0, 1.0],  # Proportionality constant [-]
    "crak": [0.0, 1.0],  # Proportionality constant [-]
    "em": [0.0, 20.0],  # Maximum plant-controlled evap rate [mm/d]
    "dsc": [1.0, 50.0],  # Maximum depression capacity [mm]
    "ads": [0.0, 1.0],  # Land fraction functioning as depression storage [-]
    "md": [0.99, 1.0],  # Depression storage parameter [-]
    "vcond": [0.0, 0.5],  # Leakage coefficient [mm/d]
    "dlev": [
        -10.0,
        10.0,
    ],  # Datum around which groundwater fluctuates relative to river bed [mm]
    "k1": [0.0, 1.0],  # Flow exchange parameter [d-1]
    "k2": [0.0, 1.0],  # Flow exchange parameter [d-1]
    "k3": [0.0, 100.0],  # Flow exchange parameter [d-1]
}

# Physical parameter descriptions.
MODHYDROLOG_PARAMS_DESC = {
    "insc": "Maximum interception capacity [mm]",
    "coeff": "Maximum infiltration loss parameter [mm]",
    "sq": "Infiltration loss exponent [-]",
    "smsc": "Maximum soil moisture capacity [mm]",
    "sub": "Proportionality constant [-]",
    "crak": "Proportionality constant [-]",
    "em": "Maximum plant-controlled evap rate [mm/d]",
    "dsc": "Maximum depression capacity [mm]",
    "ads": "Land fraction functioning as depression storage [-]",
    "md": "Depression storage parameter [-]",
    "vcond": "Leakage coefficient [mm/d]",
    "dlev": "Datum around which groundwater fluctuates relative to river bed [mm]",
    "k1": "Flow exchange parameter [d-1]",
    "k2": "Flow exchange parameter [d-1]",
    "k3": "Flow exchange parameter [d-1]",
}
def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S3 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S4 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S5 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2, S3, S4, S5


# ==============================================================================
# Sequential daily step function
# ==============================================================================

def modhydrolog_step(
    P, T, PET,
    insc, coeff, sq, smsc, sub, crak, em, dsc, ads, md, vcond, dlev, k1, k2, k3,
    S1, S2, S3, S4, S5,
    nearzero=1e-6,
):

    # ==========================================================================
    # 1. Interception store (S1).
    # ==========================================================================
    # 1.1 Evaporation.
    flux_Ei_pot = evap_1(S1, PET)
    flux_Ei = torch.minimum(flux_Ei_pot, S1)
    S1 = S1 - flux_Ei

    # 1.2 Add precipitation, then calculate hard capacity excess.
    S1 = S1 + P 
    # The newly added precipitation may temporarily raise S1 above insc.
    flux_EXC = interception_1(P, S1, insc) 
    # Limit the excess to the water currently held in S1.
    flux_EXC = torch.minimum(flux_EXC, S1)
    S1 = S1 - flux_EXC 
    
    # ==========================================================================
    # 2. Partition incoming excess among soil, interflow, recharge, and runoff.
    # ==========================================================================
    # 2.1 Infiltration.
    flux_INF = infiltration_1(coeff, sq, S2, smsc, flux_EXC)
    flux_INF = torch.minimum(flux_INF, flux_EXC)
    
    # The remaining excess becomes surface runoff.
    flux_RUN = flux_EXC - flux_INF

    # 2.2 Interflow.
    flux_INT_pot = interflow_1(sub, S2, smsc, flux_INF)
    flux_INT = torch.minimum(flux_INT_pot, flux_INF)
    
    # 2.3 Recharge.
    remain_after_int = flux_INF - flux_INT
    flux_REC_pot = recharge_1(crak, S2, smsc, remain_after_int)
    flux_REC = torch.minimum(flux_REC_pot, remain_after_int)
    
    # 2.4 Water remaining for the soil-moisture store.
    flux_SMF = remain_after_int - flux_REC
    
    # ==========================================================================
    # 3. Depression store (S3).
    # ==========================================================================
    # 3.1 Capture runoff in the depression store.
    flux_TRAP = depression_1(ads, md, S3, dsc, flux_RUN, nearzero)
    S3 = S3 + flux_TRAP
    
    # Compute surface runoff after depression capture.
    flux_SRUN = flux_RUN - flux_TRAP

    # 3.2 Evaporation from the depression store.
    flux_Ed_pot = evap_1(S3, ads * PET)
    flux_Ed = torch.minimum(flux_Ed_pot, S3)
    S3 = S3 - flux_Ed 

    # 3.3 Delayed infiltration from S3 to S2.
    flux_DINF_pot = infiltration_2(coeff, sq, S2, smsc, flux_SMF, S3)
    flux_DINF = flux_DINF_pot * ads 
    flux_DINF = torch.minimum(flux_DINF, S3) 
    S3 = S3 - flux_DINF
    
    S3 = torch.clamp(S3, min=0.0)

    # ==========================================================================
    # 4. Soil-moisture store (S2).
    # ==========================================================================
    # 4.1 Receive soil input and delayed infiltration.
    S2 = S2 + flux_SMF + flux_DINF
    
    # 4.2 Evaporation.
    flux_Et_pot = evap_2(em, S2, smsc, PET)
    flux_Et = torch.minimum(flux_Et_pot, S2)
    S2 = S2 - flux_Et
    
    # 4.3 Hard capacity excess (GWF) from the post-input, post-ET soil store.
    # This differs from MARRMoT's smooth incoming-flux saturation_1 expression.
    excess_s2 = torch.relu(S2 - smsc)
    flux_GWF = excess_s2 
    S2 = S2 - flux_GWF
    
    S2 = torch.clamp(S2, min=0.0)

    # ==========================================================================
    # 5. Groundwater and channel interaction (S4 <-> S5).
    # ==========================================================================
    # 5.1 S4 receives recharge and soil-store overflow.
    S4 = S4 + flux_REC + flux_GWF
    
    # 5.2 Add surface runoff and interflow to S5 before channel exchange.
    S5 = S5 + flux_SRUN + flux_INT
    
    # 5.3 Deep seepage from S4.
    flux_SEEP_pot = exchange_3(vcond, S4, dlev)
    # Seepage is an outward loss: a negative exchange value denotes the
    # opposite direction and must not be counted as negative evaporation.
    flux_SEEP = torch.minimum(F.relu(flux_SEEP_pot), S4)
    S4 = S4 - flux_SEEP  # Include this outward loss in the reported water loss.
    
    # 5.4 Exchange flow between S4 and S5.
    pot_flow = exchange_1(k1, k2, k3, S4, flux_SRUN)
    
    flow_out = F.relu(pot_flow)      # S4 -> S5
    flow_in  = F.relu(-pot_flow)     # S5 -> S4
    
    real_flow_out = torch.minimum(flow_out, S4)
    real_flow_in = torch.minimum(flow_in, S5)
    
    flux_FLOW = real_flow_out - real_flow_in
    
    S4 = S4 - flux_FLOW
    S5 = S5 + flux_FLOW
    
    S4 = torch.clamp(S4, min=0.0)

    # ==========================================================================
    # 6. Channel outflow from S5.
    # ==========================================================================
    flux_Q_pot = baseflow_1(torch.ones_like(S5), S5)
    flux_Q = torch.minimum(flux_Q_pot, S5)
    S5 = S5 - flux_Q
    
    S5 = torch.clamp(S5, min=0.0)

    # ==========================================================================
    # 7. Aggregate streamflow and water losses.
    # ==========================================================================
    Q = flux_Q
    Ea = flux_Ei + flux_Et + flux_Ed + flux_SEEP  # Includes seepage as an outward loss.
    
    return Q, Ea, S1, S2, S3, S4, S5
