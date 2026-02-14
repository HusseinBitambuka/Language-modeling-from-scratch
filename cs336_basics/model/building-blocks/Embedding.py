import torch
from torch import nn


class Embedding(nn.Module):

    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device|None=None, dtype:torch.dtype|None=None ) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                dtype=dtype,
                device=device
            )
        )
        self.init_param()

    def init_param(self):
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        token_ids.long()
        return self.W[token_ids]