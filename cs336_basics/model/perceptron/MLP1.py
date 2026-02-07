from cs336_basics.tokenizer import BPE2
import numpy as np
import time
from torch import embedding


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # MLP
        input_dim = context_len * d_model
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Small, stable initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        x: (batch_size, context_len)  -- token ids
        returns:
            logits: (batch_size, vocab_size)
        """

        # (batch, context_len, d_model)
        x = self.embedding(x)

        # Flatten context
        # (batch, context_len * d_model)
        x = x.view(x.size(0), -1)

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)

        return logits
    



