import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplTri4(DplUHBase):
    """Unit Hydrograph 4: full triangular density with linear rise and fall.

    Corresponds to MARRMoT ``uh_4_full``. The density peaks at
    ``d_base / 2``; its cumulative S-curve is ``2 * (t/d_base)^2`` for
    ``t <= d_base/2`` and ``1 - 2 * (1 - t/d_base)^2`` thereafter.
    Returned weights are successive cumulative increments.
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # Evaluate the cumulative curve at integer step boundaries t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)

        s1 = 2.0 * ratio.pow(2)
        term2 = torch.clamp(1.0 - ratio, min=0.0)
        s2 = 1.0 - 2.0 * term2.pow(2)
        s_curve = torch.where(ratio <= 0.5, s1, s2)
        s_curve = torch.clamp(s_curve, max=1.0)

        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights