import torch
from torch import nn
from quantizers.uniform import UniformQuantizer
from quantizers._ste import round_ste


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uq: UniformQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uq: UniformQuantizer, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid'):
        super().__init__()
        # copying all attributes from UniformQuantizer
        self.n_bits = uq.n_bits
        self.n_levels = uq.n_levels
        self.channel_wise = uq.channel_wise
        self.sym = uq.sym
        self.scale = nn.Parameter(uq.scale)
        self.zero_point = nn.Parameter(uq.zero_point)
        
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.scale)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.scale)
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels, self.n_levels - 1)
            x_float_q = x_quant * self.scale
        else:
            x_quant = torch.clamp(x_int + self.zero_point, 0, 2 * self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * self.scale
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.scale)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

    def get_hard_value(self, x):
        init_shape = x.shape
        return ((torch.floor(x.reshape_as(self.alpha) / self.scale) + (self.alpha >= 0).float()) * self.scale).reshape(*init_shape)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_bits={self.n_bits}, sym={self.sym}, channel_wise={self.channel_wise}, round_mode={self.round_mode})'
