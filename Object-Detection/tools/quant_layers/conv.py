from numpy import not_equal
import math
from torch import tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from quantizers.uniform import UniformQuantizer


class MinMaxQuantConv2d(nn.Conv2d):
    """
    MinMax quantize weight and output
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 w_bit = 8,
                 a_bit = 8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mode = mode
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = False)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        
    def forward(self, x):
        if self.mode == 'raw':
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.mode == "quant_forward":
            out=self.quant_forward(x)
        elif self.mode == 'debug_only_quant_weight':
            out = self.debug_only_quant_weight(x)
        elif self.mode == 'debug_only_quant_act':
            out = self.debug_only_quant_act(x)
        else:
            raise NotImplementedError
        return out
            
    def quant_weight_bias(self):
        w_sim = self.w_quantizer(self.weight)
        return w_sim, self.bias if self.bias is not None else None
    
    def quant_input(self,x):
        if self.a_quantizer.n_bits >= 8:
            return x
        return self.a_quantizer(x)
    
    def quant_forward(self,x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def debug_only_quant_weight(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.conv2d(x, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = F.conv2d(x_sim, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    
class PTQSLQuantConv2d(MinMaxQuantConv2d):
    """
    PTQSL on Conv2d
    weight: (oc,ic,kw,kh) -> (oc,ic*kw*kh) -> divide into sub-matrixs and quantize
    input: (B,ic,W,H), keep this shape

    Only support SL quantization on weights.
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 metric = 'mse',
                 w_bit = 8,
                 a_bit = 8,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit)
        self.metric = metric
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.search_round = search_round
        self.eq_n = eq_n
        self.parallel_eq_n = eq_n
        
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.a_quantizer.register_buffer('scale', torch.zeros((1, 1, 1, 1)))
    
    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2

    def quant_weight_bias(self):
        # self.weight_scale shape: (1, 1) or (oc, 1) 
        # self.weight       shape: (oc,ic,kw,kh)
        oc, ic, kw, kh = self.weight.data.shape
        w_sim = self.w_quantizer(self.weight.view(oc, ic * kw * kh)).view(oc, ic, kw, kh)
        return w_sim, self.bias if self.bias is not None else None

    
class PTQSLBatchingQuantConv2d(PTQSLQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 metric = 'mse',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, metric, w_bit, a_bit, search_round, eq_n)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        self.calib_size = 1
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (2 * self.raw_input[0].numel() + 
                 2 * self.raw_out[0].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
    def _initialize_activation_scale(self):
        tmp_a_scales = []
        for b_st in range(0, self.raw_input.shape[0], self.calib_batch_size):
            b_ed = min(self.raw_input.shape[0], b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].cuda()
            a_scale_=(x_.abs().max() / (self.a_quantizer.n_levels - 0.5)).detach().view(1, 1)
            tmp_a_scales.append(a_scale_)
        tmp_a_scale = torch.cat(tmp_a_scales, dim=1).amax(dim=1, keepdim=False).view(1, 1, 1, 1)
        self.a_quantizer.scale.data.copy_(tmp_a_scale) # shape: (1, 1, 1, 1)
        self.a_quantizer.inited = True
        
    def _search_best_a_scale(self, input_scale_candidates):
        batch_similarities = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                B,ic,iw,ih = x.shape
                x_sim = x.unsqueeze(0) # shape: 1,B,ic,iw,ih
                x_sim = (x_sim / (cur_a_scale)).round_().clamp_(-self.a_quantizer.n_levels, self.a_quantizer.n_levels - 1) * cur_a_scale # shape: parallel_eq_n,B,ic,iw,ih
                x_sim = x_sim.view(-1,ic,iw,ih)
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: parallel_eq_n*B,oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(0), chunks=p_ed-p_st, dim=1), dim=0) # shape: parallel_eq_n,B,oc,fw,fh
                out_sim = out_sim.transpose_(0, 1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, dim=[2,3,4]) # shape: B,parallel_eq_n
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1,parallel_eq_n
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: eq_n
        best_index = batch_similarities.argmax(dim=0).view(1,1,1,1,1)
        tmp_a_scale = torch.gather(input_scale_candidates, dim=0, index=best_index)
        self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(0))
        
        
class AsymmetricallyBatchingQuantConv2d(PTQSLBatchingQuantConv2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 mode = 'raw',
                 metric = 'mse',
                 w_bit = 8,
                 a_bit = 8,
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, metric, w_bit, a_bit, calib_batch_size, search_round, eq_n)
        
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = False, channel_wise = False)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        self.a_quantizer.zero_point = nn.Parameter(torch.zeros((1, 1, 1, 1)))
    
    def _search_best_w_scale_self(self, weight_scale_candidates, weight_zero_point_candidates, topk=1):
        similarities = []
        for p_st in range(0, self.eq_n, self.parallel_eq_n):
            p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
            cur_w_scale = weight_scale_candidates[p_st:p_ed] # shape: (parallel_eq_n, 1, 1) or (parallel_eq_n, oc, 1)
            cur_w_zero_point = weight_zero_point_candidates[p_st:p_ed]
            # quantize weight and bias 
            oc,ic,kw,kh = self.weight.data.shape
            w_sim = self.weight.view(oc, -1).unsqueeze(0) # shape: (1, oc, ic*kw*kh)
            w_quant = ((w_sim / cur_w_scale).round_() + cur_w_zero_point).clamp(0, 2 * self.w_quantizer.n_levels - 1)
            w_sim = (w_quant - cur_w_zero_point).mul_(cur_w_scale) # shape: (parallel_eq_n,oc,ic*kw*kh)
            w_sim = w_sim.view(-1,oc,ic,kw,kh) # shape: parallel_eq_n,oc,ic,kw,kh
            similarity = self._get_similarity(self.weight, w_sim) # shape: parallel_eq_n,oc,ic,kw,kh
            similarity = torch.mean(similarity, [2,3,4]) # shape: parallel_eq_n,oc
            similarities.append(similarity)
        similarities = torch.cat(similarities, dim=0) # shape: (eq_n,1) or (eq_n,oc)
        _, best_index = torch.topk(similarities, k=topk, dim=0)
        best_index = best_index.view(topk, -1, 1)
        if topk == 1:
            tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
            tmp_w_zero_point = torch.gather(weight_zero_point_candidates, dim=0, index=best_index)
            self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
            self.w_quantizer.zero_point.data.copy_(tmp_w_zero_point.squeeze(dim=0))
        return best_index
    
    def _search_best_a_scale_self(self, input_scale_candidates, input_zero_point_candidates, topk=1):
        batch_similarities = []
        for f in range(len(self.raw_input)):
            x = self.raw_input[f].cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_scale = input_scale_candidates[p_st:p_ed]
                cur_a_zero_point = input_zero_point_candidates[p_st:p_ed]
                # quantize input
                B,ic,iw,ih = x.shape
                x_sim = x.unsqueeze(0) # shape: 1,1,ic,iw,ih
                x_quant = ((x_sim / cur_a_scale).round() + cur_a_zero_point).clamp_(0, 2 * self.a_quantizer.n_levels - 1)
                x_dequant = (x_quant - cur_a_zero_point).mul_(cur_a_scale) # shape: parallel_eq_n,1,ic,iw,ih
                # calculate similarity and store them
                similarity = self._get_similarity(x.unsqueeze(0), x_dequant)
                similarity = torch.mean(similarity, dim=[1,2,3,4]) # shape: parallel_eq_n
                similarities.append(similarity.unsqueeze(0))
            similarities = torch.cat(similarities, dim=1) # shape: 1,eq_n
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: eq_n
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, 1, 1, 1)
        if topk == 1:
            tmp_a_scale = torch.gather(input_scale_candidates, dim=0, index=best_index)
            tmp_a_zero_point = torch.gather(input_zero_point_candidates, dim=0, index=best_index)
            self.a_quantizer.scale.data.copy_(tmp_a_scale.squeeze(dim=0))
            self.a_quantizer.zero_point.data.copy_(tmp_a_zero_point.squeeze(dim=0))
        return best_index
    
    def calculate_percentile_weight_candidates(self, l=0.999, r=1.0):
        num_zp = self.w_quantizer.n_levels
        num_scale = int(self.eq_n / num_zp)
        pct = torch.tensor([l, r])
        w_uppers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: 2, out_channels, 1
        w_lowers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), (1-pct).to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shapeL 2, out_channels, 1
        delta_min = w_uppers_candidates[0:1] - w_lowers_candidates[0:1]
        delta_max = w_uppers_candidates[1:] - w_lowers_candidates[1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None] * (delta_max - delta_min)
        weight_scale_candidates = (delta_min + splits).repeat(num_zp, 1, 1) / (2 * self.w_quantizer.n_levels - 1)
        zp_min = int(self.w_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.w_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        weight_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None]
        weight_zero_point_candidates = weight_zero_point_candidates.repeat(1, self.out_channels, 1)
        return weight_scale_candidates, weight_zero_point_candidates
    
    def calculate_percentile_activation_candidates(self, l=0.999, r=1.0):
        num_zp = self.a_quantizer.n_levels
        num_scale = int(self.eq_n / num_zp)
        percentiles_uppers, percentiles_lowers = [], []
        pct = torch.tensor([l, r])
        x = torch.cat([_.cuda().view(-1) for _ in self.raw_input])
        tensor_too_large = True
        mini_batch_size = 1
        while tensor_too_large:
            try:
                a_uppers_candidates = torch.quantile(x.view(mini_batch_size, -1), pct.to(x.device), dim=-1).mean(dim=-1).view(-1, 1, 1, 1, 1) # shape: 2,1,1,1,1
                a_lowers_candidates = torch.quantile(x.view(mini_batch_size, -1), (1-pct).to(x.device), dim=-1).mean(dim=-1).view(-1, 1, 1, 1, 1) # shape: 2,1,1,1,1
                tensor_too_large = False
            except:
                mini_batch_size *= 2
        delta_min = a_uppers_candidates[0:1] - a_lowers_candidates[0:1]
        delta_max = a_uppers_candidates[1:] - a_lowers_candidates[1:]
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None, None, None] * (delta_max - delta_min)
        a_scale_candidates = ((delta_min + splits).repeat(num_zp, 1, 1, 1, 1) / (2 * self.a_quantizer.n_levels - 1)).clamp(min=1e-4)
        zp_min = int(self.a_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.a_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        a_zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None, None, None]
        return a_scale_candidates, a_zero_point_candidates
    
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        
        weight_scale_candidates, weight_zero_point_candidates = self.calculate_percentile_weight_candidates()
        self.w_quantizer.scale.data.copy_(weight_scale_candidates[-2])
        self.w_quantizer.zero_point.data.copy_(weight_zero_point_candidates[-2])
        self.w_quantizer.inited = True
        if self.a_quantizer.n_bits < 8:
            input_scale_candidates, input_zero_point_candidates = self.calculate_percentile_activation_candidates()
            self.a_quantizer.scale.data.copy_(input_scale_candidates[-2])
            self.a_quantizer.zero_point.data.copy_(input_zero_point_candidates[-2])
            self.a_quantizer.inited = True
            
        self._search_best_w_scale_self(weight_scale_candidates, weight_zero_point_candidates)
        if self.a_quantizer.n_bits < 8:
            self._search_best_a_scale_self(input_scale_candidates, input_zero_point_candidates)

        self.calibrated = True
        del self.raw_input, self.raw_out
            