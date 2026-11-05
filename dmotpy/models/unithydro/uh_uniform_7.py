import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplUniform7(DplUHBase):
    """Unit Hydrograph 7: uniform density over ``d_base``.

    Corresponds to MARRMoT ``uh_7_uniform``. The cumulative S-curve is
    ``min(t / d_base, 1)``; returned weights are successive increments.
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # Evaluate cumulative increments at integer step boundaries t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)
        s_curve = torch.clamp(ratio, max=1.0)
        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights