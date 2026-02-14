import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 device=None, dtype=None):
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(normalized_shape, device=device, dtype=dtype)
        )

        self.bias = nn.Parameter(
            torch.zeros(normalized_shape, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean over last dimension
        mean = x.mean(dim=-1, keepdim=True)

        # Compute variance over last dimension
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        return self.weight * x_hat + self.bias
