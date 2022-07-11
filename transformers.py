import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_pos=2048):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe = torch.view_as_real(
            torch.exp(1j * position * div_term)).view(max_pos, d_embed)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        return self.pe[:seq_len].clone()
