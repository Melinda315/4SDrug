import torch
import torch.nn as nn
import numpy as np
import math


class Attention(nn.Module):
    def __init__(self, embed_dim=64, output_dim=1):
        super(Attention, self).__init__()
        self.embedding_dim, self.output_dim = embed_dim, output_dim
        self.aggregation = nn.Linear(self.embedding_dim, self.output_dim)

    def _aggregate(self, x):
        weight = self.aggregation(x)  # [b, num_learn, 1]
        return torch.tanh(weight)

    def forward(self, x, mask=None):
        if mask is None:
            weight = torch.softmax(self._aggregate(x), dim=-2)
        else:
            device = mask.device
            mask = torch.where(mask == 0, torch.tensor(-1e7).to(device), torch.tensor(0.0).to(device))
            weight = torch.softmax(self._aggregate(x).squeeze(-1) + mask, dim=-1).float().unsqueeze(-1)
            weight = torch.where(torch.isnan(weight), torch.tensor(0.0).to(device), weight)
        agg_embeds = torch.matmul(x.transpose(-1, -2).float(), weight).squeeze(-1)
        return agg_embeds

