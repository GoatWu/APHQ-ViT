from numpy import not_equal
import math
from torch import tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from quantizers.uniform import *


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
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit)
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = True, channel_wise = True)
        self.a_quantizer = UniformQuantizer(n_bits = a_bit, symmetric = True, channel_wise = False)
        self.metric = metric
        self.search_round = search_round
        self.eq_n = eq_n
        self.parallel_eq_n = eq_n
        
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.a_quantizer.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.a_quantizer.register_buffer('scale', torch.zeros((1, 1, 1, 1)))
    
    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        if metric == "mae":
            similarity = -torch.abs(tensor_raw - tensor_sim)
        elif metric == "mse":
            similarity = -(tensor_raw - tensor_sim) ** 2
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        return similarity

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
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, metric, search_round, eq_n)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = self.raw_input.shape[0]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (2 * self.raw_input[:self.calib_batch_size].numel() + 
                 2 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
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
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: B,parallel_eq_n,oc,fw,fh
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
                 w_bit = 8,
                 a_bit = 8,
                 metric = "mse", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
                         bias, padding_mode, mode, w_bit, a_bit, metric, calib_batch_size, search_round, eq_n)

        del self.w_quantizer
        self.w_quantizer = UniformQuantizer(n_bits = w_bit, symmetric = False, channel_wise = True)
        self.w_quantizer.scale = nn.Parameter(torch.zeros((self.out_channels, 1)))
        self.w_quantizer.zero_point = nn.Parameter(torch.zeros((self.out_channels, 1)))
    
    def _search_best_w_scale(self, weight_scale_candidates, weight_zero_point_candidates):
        batch_similarities = []
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].cuda()
            raw_out = self.raw_out[b_st:b_ed].cuda().unsqueeze(1) # shape: b,1,oc,fw,fh
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
                w_sim = w_sim.view(-1,ic,kw,kh) # shape: parallel_eq_n*oc,ic,kw,kh
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # calculate similarity and store them
                out_sim = F.conv2d(x_sim, w_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups) # shape: B,parallel_eq_n*oc,fw,fh
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(1), chunks=p_ed-p_st, dim=2), dim=1) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = self._get_similarity(raw_out, out_sim, self.metric) # shape: B,parallel_eq_n,oc,fw,fh
                similarity = torch.mean(similarity, [3,4]) # shape: B,parallel_eq_n,oc
                similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: (1,parallel_eq_n) or (1,parallel_eq_n,oc)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=1) # shape: (1,eq_n) or (1,eq_n,oc)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) #shape: (eq_n) or (eq_n,oc)
        best_index = batch_similarities.argmax(dim=0).reshape(1, -1, 1) # shape: (1,1,1) or (1,oc,1)
        tmp_w_scale = torch.gather(weight_scale_candidates, dim=0, index=best_index)
        tmp_w_zero_point = torch.gather(weight_zero_point_candidates, dim=0, index=best_index)
        self.w_quantizer.scale.data.copy_(tmp_w_scale.squeeze(dim=0))
        self.w_quantizer.zero_point.data.copy_(tmp_w_zero_point.squeeze(dim=0))
        return best_index
    
    def calculate_percentile_weight_candidates(self, l=0.99, r=0.9999, k=0.05):
        pct = torch.tensor([l + (r - l) * (i / (self.eq_n - 1))**k for i in range(self.eq_n)] + [1.0])
        w_uppers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), pct.to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shape: eq_n, out_channels, 1
        w_lowers_candidates = torch.quantile(
            self.weight.view(self.out_channels, -1), (1-pct).to(self.weight.device), dim=-1
        ).unsqueeze(-1) # shapeL eq_n, out_channels, 1
        return w_uppers_candidates, w_lowers_candidates

    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        self._initialize_activation_scale()
        self.eq_alpha, self.eq_beta = 0.01, 1.2
        input_scale_candidates =  torch.tensor(
            [self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]
        ).cuda().view(-1,1,1,1,1) * self.a_quantizer.scale # shape: (eq_n,1,1,1,1)
        
        w_uppers_candidates, w_lowers_candidates = self.calculate_percentile_weight_candidates(l=0.99, r=0.9999, k=0.5)
        weight_scale_candidates = ((w_uppers_candidates - w_lowers_candidates) / (2 * self.w_quantizer.n_levels - 1)).contiguous().cuda()
        weight_zero_point_candidates = -(w_lowers_candidates / weight_scale_candidates).round().contiguous().cuda()
        w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
        self.w_quantizer.inited = True
        
        for e in range(self.search_round):
            for ee in range(2):
                if ee % 2 == 0:
                    w_uppers_candidates_ = torch.gather(w_uppers_candidates, dim=0, index=w_best_index)
                    w_lowers_candidates_ = w_lowers_candidates
                else:
                    w_uppers_candidates_ = w_uppers_candidates
                    w_lowers_candidates_ = torch.gather(w_lowers_candidates, dim=0, index=w_best_index)
                weight_scale_candidates = ((w_uppers_candidates_ - w_lowers_candidates_) / (2 * self.w_quantizer.n_levels - 1)).contiguous().cuda()
                weight_zero_point_candidates = -(w_lowers_candidates_ / weight_scale_candidates).round().contiguous().cuda()
                w_best_index = self._search_best_w_scale(weight_scale_candidates, weight_zero_point_candidates)
            if self.a_quantizer.n_bits < 8:
                self._search_best_a_scale(input_scale_candidates)
            else:
                break
        self.calibrated = True
        del self.raw_input, self.raw_out
            