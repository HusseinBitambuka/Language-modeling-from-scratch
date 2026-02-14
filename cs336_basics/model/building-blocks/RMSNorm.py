import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        # Upcast for numerical stability
        x = x.to(torch.float32)

        # Compute mean square
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Compute RMS
        rms = torch.sqrt(mean_square + self.eps)

        # Normalize and apply gain
        result = (x / rms) * self.gain

        # Cast back to original dtype
        return result.to(in_dtype)


