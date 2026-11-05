import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplDelay8(DplUHBase):
    """Unit Hydrograph 8: fractional pure delay (MARRMoT ``uh_8_delay``).

    The delay is clamped to nonnegative values. A fractional delay distributes
    weight between its two adjacent integer lag positions using the
    piecewise-linear kernel ``relu(1 - |lag - (delay + 1)|)``. The kernel has
    nonsmooth boundaries at its ReLU and absolute-value breakpoints.
    """
    
    def get_weights(self, params):
        t_delay = torch.clamp(params, min=0.0)

        # This delay kernel uses integer lag positions rather than the base half-step grid.
        t_idx = torch.arange(
            1, self.max_lag + 1, device=t_delay.device, dtype=t_delay.dtype
        ).view(1, 1, -1)

        center = t_delay + 1.0
        dist = torch.abs(t_idx - center.unsqueeze(-1))
        weights = F.relu(1.0 - dist)
        return weights
