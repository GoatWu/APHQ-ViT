import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers._ste import *


class UniformQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.channel_wise = channel_wise
        self.drop_prob = 1.0
        self.inited = False
        self.training_mode = False

    def init_training(self):
        self.training_mode = True

    def end_training(self):
        self.training_mode = False
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x
        x_int = round_ste(x / self.scale) if self.training_mode else torch.round(x / self.scale)
        if self.sym:
            x_quant = x_int.clamp(-self.n_levels, self.n_levels - 1)
            x_dequant = x_quant * self.scale
        else:
            x_quant = (x_int + torch.round(self.zero_point)).clamp(0, 2 * self.n_levels - 1)
            x_dequant = (x_quant - torch.round(self.zero_point)) * self.scale
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise})'
