import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from quantizers._ste import *


class Log2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** (self.n_bits - 1)
        self.inited = False
        self.drop_prob = 1.0
        self.channel_wise = channel_wise
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
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        x_quant = round_ste(-scaled_x.log2()) if self.training_mode else torch.round(-scaled_x.log2())
        mask = x_quant < 2 * self.n_levels
        x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
        x_dequant = 2 ** (-1 * x_quant) * self.scale
        x_dequant = x_dequant * mask
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, log_base={2})'


class LogSqrt2Quantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        
    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        if self.training_mode:
            x_quant = round_ste(-scaled_x.log2() * 2)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = 2 ** (-1 * x_quant / 2) * self.scale
        else:
            x_quant = torch.round(-scaled_x.log2() * 2)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
            x_dequant = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * self.scale
        x_dequant = x_dequant * mask
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, log_base={math.sqrt(2)})'
    
    
class AdaLogQuantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.r = 37.0
        self.register_buffer('q', torch.tensor([int(self.r)]))
        self.register_buffer('table', torch.zeros((self.n_levels * 2)))
        self.update_table()

    def update_table(self):
        for i in range(0, self.n_levels * 2):
            val = round((2 ** (-((self.q.item() * i) % self.r) / self.r)) * (4 * self.n_levels - 2)) / (4 * self.n_levels - 2)
            self.table[i].data.copy_(torch.tensor(val))

    def init_training(self):
        self.training_mode = True
        self.q = self.q.to(torch.float32)

    def end_training(self):
        self.training_mode = False
        self.q.data.copy_(torch.round(self.q))
        self.update_table()

    def forward(self, x):
        if self.n_bits == 32:
            return x
        assert self.inited
        if self.training_mode and self.drop_prob < 1.0:
            x_orig = x
        scaled_x = (x / self.scale).clamp(min=1e-15, max=1.0)
        if self.training_mode:
            x_quant = round_ste(-scaled_x.log2() * self.r / self.q)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = 2 ** (-1 * x_quant * self.q / self.r) * self.scale
        else:
            x_quant = torch.round(-scaled_x.log2() * self.r / self.q)
            mask = x_quant < 2 * self.n_levels
            x_quant = torch.clamp(x_quant, 0, 2 * self.n_levels - 1)
            x_dequant = (2 ** (-1 * torch.floor(x_quant * self.q / self.r))) * self.table[x_quant.long()] * self.scale
        x_dequant = x_dequant * mask
        if self.training_mode and self.drop_prob < 1.0:
            x_prob = torch.where(torch.rand_like(x) < self.drop_prob, x_dequant, x_orig)
            return x_prob
        return x_dequant

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, q={self.q.item()})'
        

class ShiftLog2Quantizer(Log2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        result = Log2Quantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift


class ShiftLogSqrt2Quantizer(LogSqrt2Quantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        result = LogSqrt2Quantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift


class ShiftAdaLogQuantizer(AdaLogQuantizer):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False):
        super().__init__(n_bits, symmetric, channel_wise)
        self.shift = nn.Parameter(torch.zeros((1)))
        self.register_buffer('bias_reparamed', torch.tensor(False))

    def forward(self, x):
        result = AdaLogQuantizer.forward(self, x + self.shift)
        return result if self.bias_reparamed else result - self.shift
        