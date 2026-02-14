import torch
import torch.nn as nn
import math


class Linear(nn.Module):

    def __init__(self, in_features:int, out_features:int, device:torch.device|None=None, dtype:torch.dtype|None=None) ->None:
        """
        linear transformation module. This function should accept the following parameters:

        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features

        self.W = nn.Parameter(
            torch.empty(
                in_features,
                out_features,
                device=device,
                dtype=dtype
            )
        )
        self.init_param()

    def init_param(self):
        sigma:float = math.sqrt(2.0/(self.d_in + self.d_out))
        torch.nn.init.trunc_normal_(self.W, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, X:torch.Tensor) -> torch.Tensor:

        return X @ self.W
