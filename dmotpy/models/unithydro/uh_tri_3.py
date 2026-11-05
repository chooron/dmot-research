import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplTri3(DplUHBase):
    """Unit Hydrograph 3: half triangle with a linearly rising density.

    Corresponds to MARRMoT ``uh_3_half``. Its cumulative S-curve is
    ``(t / d_base)^2`` for ``t <= d_base`` and one thereafter; weights are
    successive cumulative increments, not linearly increasing weights.
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # Evaluate cumulative increments at integer step boundaries t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0).pow(2.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights