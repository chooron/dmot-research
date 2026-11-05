import torch
import torch.nn.functional as F

def recharge_1(p1, S, Smax, flux, nearzero=1e-6):
    """
    Recharge as scaled fraction of incoming flux
    Formula: out = p1 * S / Smax * flux
    """
    Smax_safe = torch.clamp(Smax, min=1.0)
    ratio = torch.clamp(S / Smax_safe, max=1.0)
    return p1 * ratio * flux

def recharge_2(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Recharge as non-linear scaling of incoming flux
    Formula: out = flux * (max(S,0) / Smax)^p1
    """
    ratio = torch.clamp(F.relu(S) / (Smax + nearzero), max=1.5)
    return flux * (ratio + nearzero).pow(p1)


def recharge_3(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Linear recharge
    Formula: out = p1 * S
    """
    return p1 * S


def recharge_4(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant recharge
    Formula: out = min(p1, S)
    Note: dt is assumed to be 1.0
    """
    return torch.minimum(p1, S)


def recharge_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Recharge scaled by a softplus of the receiving-store deficit.

    The implementation lower-bounds ``p2`` at 1.0 and applies
    ``softplus(1 - S2 / p2_safe, beta=10)`` without an upper cap. The softplus
    therefore does not equal the hard positive-part reference and can exceed
    one for a sufficiently large deficit.
    """
    # Keep the denominator positive for externally supplied parameter values.
    p2_safe = torch.clamp(p2, min=1.0)
    
    # Positive diff denotes storage below the threshold; negative diff denotes excess.
    diff = 1.0 - S2 / p2_safe
    
    # Softplus replaces the hard positive-part operator; beta controls its scale.
    term = F.softplus(diff, beta=10.0)
    
    # No upper clamp is applied: large positive deficits can produce term > 1.
    return p1 * S1 * term


def recharge_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Recharge to fulfil evaporation demand if the receiving store is below a threshold
    Formula: out = min(max(S,0), p1 * max(S,0)^p2)
    Note: dt is assumed to be 1.0
    """
    s_pos = F.relu(S)
    potential = p1 * (s_pos + nearzero).pow(p2)
    return torch.minimum(s_pos, potential)


def recharge_7(
    p1: torch.Tensor, incoming_flux: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Constant recharge limited by incoming flux
    Formula: out = min(p1, fin)
    """
    return torch.minimum(p1, incoming_flux)


def recharge_8(
    p1: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    p2: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Recharge as non-linear scaling of incoming flux
    Formula: out = min(p2 * (max(S,0)/Smax)^p1, max(S,0))
    Note: dt is assumed to be 1.0
    """
    ratio = F.relu(S) / (Smax + nearzero)
    potential = p2 * (ratio + nearzero).pow(p1)
    return torch.minimum(potential, F.relu(S))
