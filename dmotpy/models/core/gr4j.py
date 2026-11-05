import torch
import torch.nn.functional as F
from typing import Tuple
from ..flux.exchange import exchange_gr4j

# GR4J production-store and analytical routing helper functions.

def _calc_production_store_tanh(
    S: torch.Tensor, x1: torch.Tensor, Pn: torch.Tensor, En: torch.Tensor, nearzero: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute production-store rainfall and evaporation fluxes with tanh terms."""
    ratio_s_x1 = S / (x1 + nearzero)
    
    # Ps
    tanh_pn_x1 = torch.tanh(Pn / (x1 + nearzero))
    ps_num = x1 * (1.0 - ratio_s_x1.pow(2)) * tanh_pn_x1
    ps_den = 1.0 + ratio_s_x1 * tanh_pn_x1
    ps = ps_num / (ps_den + nearzero)
    
    # Es
    tanh_en_x1 = torch.tanh(En / (x1 + nearzero))
    es_num = S * (2.0 - ratio_s_x1) * tanh_en_x1
    es_den = 1.0 + (1.0 - ratio_s_x1) * tanh_en_x1
    es = es_num / (es_den + nearzero)
    
    return ps, es

def _calc_percolation_analytical(
    S: torch.Tensor, x1: torch.Tensor, nearzero: float
) -> torch.Tensor:
    """Compute the analytical-form percolation flux used by this implementation."""
    ratio_perc = (4.0 / 9.0) * (S / (x1 + nearzero))
    term_perc = (1.0 + ratio_perc.pow(4)).pow(-0.25)
    perc = S * (1.0 - term_perc)
    return perc

def _calc_routing_outflow_analytical(
    S2: torch.Tensor, x3: torch.Tensor, nearzero: float
) -> torch.Tensor:
    """Compute the analytical-form routing-store outflow used by this implementation."""
    ratio_s2_x3 = S2 / (x3 + nearzero)
    term_qr = (1.0 + ratio_s2_x3.pow(4)).pow(-0.25)
    qr = S2 * (1.0 - term_qr)
    return qr

# Daily GR4J state and routing updates.

def gr4j_step_pre(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    x1: torch.Tensor,
    S1: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GR4J pre-UH step: production store (S1).

    Returns:
        flux_pr      -- effective rainfall entering routing (enters UH)
        e_physical   -- physical evaporation (passthrough to post)
        S1_new       -- updated production store
    """

    # 1. Partition precipitation and PET into net precipitation, net evaporation, and intercepted evaporation.
    diff = P - PET
    flux_pn = F.relu(diff)      # Net precipitation.
    flux_en = F.relu(-diff)     # Net evaporation demand.
    flux_ei = P - flux_pn       # Interception evaporation.

    nearzero_tensor = torch.zeros_like(flux_pn) + nearzero
    # Keep production storage within its lower and upper bounds.
    S1 = torch.clamp(S1, min=nearzero_tensor, max=x1)

    # 2. Production-store fluxes and state update.
    flux_ps, flux_es = _calc_production_store_tanh(S1, x1, flux_pn, flux_en, nearzero)

    # Update storage after production-store evaporation and rainfall.
    S1_mid = S1 - flux_es + flux_ps
    S1_mid = torch.clamp(S1_mid, min=nearzero_tensor, max=x1)

    # Percolation from the production store.
    flux_perc = _calc_percolation_analytical(S1_mid, x1, nearzero)
    S1_new = S1_mid - flux_perc
    S1_new = torch.clamp(S1_new, min=nearzero_tensor, max=x1)

    # Effective rainfall passed to routing.
    flux_pr = flux_perc + (flux_pn - flux_ps)
    e_physical = flux_ei + flux_es

    return flux_pr, e_physical, S1_new


def gr4j_step_post(
    flux_q9: torch.Tensor,
    flux_q1: torch.Tensor,
    S2: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    e_physical: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GR4J post-UH step: routing store (S2) with exchange F and water balance.

    Args:
        flux_q9     -- 90% inflow (UH-routed or direct)
        flux_q1     -- 10% inflow (UH-routed or direct)
        S2          -- routing store
        x2, x3      -- routing parameters
        e_physical  -- physical evaporation (from pre)

    Returns:
        Qsim, Ea_balanced, S2_new
    """

    S2 = torch.clamp(S2, min=nearzero)

    # 4. Routing-store update with the external exchange term.
    # Potential exchange flux.
    flux_f_theoretical = exchange_gr4j(x2, S2, x3, nearzero=nearzero)

    # Integrate routed inflow and potential exchange, then apply the nonnegative state bound.
    S2_before_exchange = S2
    S2_integrated = S2 + flux_q9 + flux_f_theoretical
    S2_integrated = torch.clamp(S2_integrated, min=nearzero)

    # Recover the exchange realized after the storage bound.
    f_actual_s2 = (S2_integrated - S2_before_exchange) - flux_q9

    # Routing-store outflow.
    flux_qr = _calc_routing_outflow_analytical(S2_integrated, x3, nearzero)
    S2_new = S2_integrated - flux_qr

    # 5. Direct-branch flow including potential exchange.
    # Qd = relu(Q1 + F).
    flux_qd_potential = flux_q1 + flux_f_theoretical
    flux_qd = F.relu(flux_qd_potential)

    # Recover the direct-branch exchange implied by the nonnegative-flow bound.
    f_actual_q1 = flux_qd - flux_q1

    # 6. Aggregate discharge and report the balance-adjusted ET term.
    Qsim = flux_qr + flux_qd

    # Positive exchange is a gain to the modeled stores; negative exchange is a loss.
    F_total_actual = f_actual_s2 + f_actual_q1

    # This output subtracts realized exchange from physical ET for the reported balance term.
    Ea_balanced = e_physical - F_total_actual

    return Qsim, Ea_balanced, S2_new


def gr4j_step(
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one daily GR4J core step with balance-adjusted ET.

    The reported ET subtracts the exchange realized by the routing and direct branches.
    """

    # UH parameter is unused (identity routing)
    _ = x4

    flux_pr, e_physical, S1_new = gr4j_step_pre(
        P, T, PET, x1, S1, nearzero,
    )
    flux_q9 = 0.9 * flux_pr
    flux_q1 = 0.1 * flux_pr
    Qsim, Ea_balanced, S2_new = gr4j_step_post(
        flux_q9, flux_q1, S2, x2, x3, e_physical, nearzero,
    )
    return Qsim, Ea_balanced, S1_new, S2_new


# Parameter range dictionary (based on MARRMoT m_07_gr4j_4p_2s)
GR4J_PARAMS_BOUNDS = {
    "x1": [1.0, 2000.0],  # Max soil moisture storage [mm]
    "x2": [-20.0, 20.0],  # Water exchange coefficient [mm/d]
    "x3": [1.0, 300.0],   # Max routing store storage [mm]
    "x4": [0.5, 15.0],    # Flow delay [d]
}



def create_initial_state(
    n_grid: int, nmul: int, device: torch.device, nearzero: float = 1e-6
) -> Tuple[
    torch.Tensor, torch.Tensor
]:
    """
    Create initial production-store (S1) and routing-store (S2) states for GR4J.
    """
    S1 = torch.zeros((n_grid, nmul), device=device) + nearzero
    S2 = torch.zeros((n_grid, nmul), device=device) + nearzero
    return S1, S2
