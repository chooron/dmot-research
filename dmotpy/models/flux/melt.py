import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def melt_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    T: torch.Tensor,
    S: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Snowmelt from degree-day-factor.
    Formula: out = max(min(p1 * (T - p2), S), 0)
    Note: dt is assumed to be 1.0.
    """
    melt_potential = p1 * (T - p2)
    return F.relu(torch.minimum(melt_potential, S))


def melt_2(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Snowmelt at a constant rate.
    Formula: out = min(p1, S)
    Note: dt is assumed to be 1.0.
    """
    return torch.minimum(p1, S)


def melt_3(
    p1: torch.Tensor,
    p2: torch.Tensor,
    T: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    St: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Glacier melt from the ice store S1, active only while the snow store S2 is depleted.

    ``melt_actual`` follows MARRMoT ``melt_1``: the degree-day potential
    ``p1 * (T - p2)`` cut off at zero and limited by the available ice ``S1``.
    The flux is then gated by ``1 - smooth_threshold_storage_logistic(S2, St)``.

    Note on the gate: the dMoT helper of that name is an *above*-threshold gate
    (it tends to 1 for ``S2 > St``), whereas MARRMoT's function of the same name
    is a *below*-threshold gate. MARRMoT ``melt_3`` therefore writes
    ``... .* smoothThreshold_storage_logistic(S2, St)`` directly, and the extra
    ``1 -`` here reproduces that behaviour. The gate centre is at ``St`` with a
    different transition width from MARRMoT (see the formula-difference audit).

    Note: ``delta_t`` is assumed to be 1.0, so the MARRMoT ``S/dt`` divisors
    reduce to the plain storage values used here.
    """
    melt_potential = F.relu(p1 * (T - p2))
    melt_actual = torch.minimum(melt_potential, S1)
    sf = smooth_threshold_storage_logistic(S2, St, nearzero=nearzero)
    return melt_actual * (1.0 - sf)
