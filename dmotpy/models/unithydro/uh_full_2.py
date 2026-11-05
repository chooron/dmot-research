import torch
import torch.nn.functional as F
from .base import DplUHBase

class DplFull2(DplUHBase):
    """GR4J full bell-curve unit hydrograph (MARRMoT ``uh_2_full``).

    Its cumulative S-curve spans ``2 * d_base``: ``0.5 * (t/d_base)^2.5``
    for ``t <= d_base`` and ``1 - 0.5 * (2 - t/d_base)^2.5`` thereafter,
    capped at one. Returned weights are successive S-curve increments.
    """
    
    def get_weights(self, params):
        d_base = torch.clamp(params, min=1e-3)

        # Evaluate cumulative increments at integer step boundaries t=1,2,...
        t_idx = torch.arange(
            1, self.max_lag + 1, device=d_base.device, dtype=d_base.dtype
        ).view(1, 1, -1)

        ratio = t_idx / d_base.unsqueeze(-1)

        s_part1 = 0.5 * ratio.pow(2.5)
        term_b = torch.clamp(2.0 - ratio, min=0.0)
        s_part2 = 1.0 - 0.5 * term_b.pow(2.5)
        s_curve = torch.where(ratio <= 1.0, s_part1, s_part2)
        s_curve = torch.clamp(s_curve, max=1.0)

        s_curve_padded = F.pad(s_curve, (1, 0), value=0.0)
        weights = s_curve - s_curve_padded[..., :-1]

        return weights