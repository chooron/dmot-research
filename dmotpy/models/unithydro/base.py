import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DplUHBase(nn.Module, ABC):
    """Unit-hydrograph convolution base using grouped ``Conv1d``.

    The base buffer samples half-step times for subclasses that use it; a
    subclass may supply its own sampling grid. Weights are normalized with an
    epsilon denominator, reversed for PyTorch cross-correlation, and cropped
    after symmetric padding to retain the input sequence length.
    """

    def __init__(self, max_lag, epsilon=1e-6):
        super().__init__()
        self.max_lag = int(max_lag)
        self.epsilon = epsilon
        
        self.register_buffer(
            "t_idx",
            torch.arange(0.5, self.max_lag * 1.0, dtype=torch.float32).view(
                1, 1, -1
            ),
        )

    @abstractmethod
    def get_weights(self, params):
        """Subclass implements specific distribution, returning unnormalized weights.
        Shape: [batch_size, 1, max_lag]
        """
        raise NotImplementedError

    def forward(self, flux_in, params):
        if params.dim() == 1:
            params = params.unsqueeze(-1)
        
        batch_size, time_steps = flux_in.shape

        raw_weights = self.get_weights(params)
        
        sum_w = raw_weights.sum(dim=-1, keepdim=True)
        denom = torch.where(
            sum_w > self.epsilon,
            sum_w,
            torch.full_like(sum_w, self.epsilon),
        )
        norm_weights = raw_weights / denom

        flipped_weights = torch.flip(norm_weights, dims=[-1])

        x = flux_in.view(1, batch_size, time_steps)
        
        padd = self.max_lag - 1

        flux_out = F.conv1d(
            input=x, 
            weight=flipped_weights, 
            groups=batch_size,
            padding=padd 
        )

        if padd > 0:
            flux_out = flux_out[:, :, 0:-padd]

        return flux_out.view(batch_size, time_steps)
