import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from timm.models.swin_transformer import window_partition, window_reverse
from utils.calibrator import QuantCalibrator
from quantizers._ste import *
from quantizers.adaround import AdaRoundQuantizer
from quantizers.uniform import UniformQuantizer
from quant_layers import *
from types import MethodType
import logging
import random
import copy


def mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    x = self.drop2(x)
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-6
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-6
    return x


def positive_percentile(tensor, pct):
    mini_batch_size = 1
    tensor_too_large = True
    while tensor_too_large:
        try:
            t = tensor.view(mini_batch_size, -1)[0:1, :]
            t = t.view(-1)
            positive_mask = t > 0
            positive_tensor = torch.where(positive_mask, t, torch.tensor(float('nan')).to(t.device))
            sorted_tensor, _ = positive_tensor.sort(dim=0)
            tensor_too_large = False
        except:
            mini_batch_size *= 2
    counts = (~torch.isnan(sorted_tensor)).sum(dim=0, keepdim=True).float()
    ranks = ((counts * pct).ceil().long() - 1).clamp(min=0)
    result = torch.gather(sorted_tensor, 0, ranks).squeeze()
    return result.item()


class MLPReconstructor(QuantCalibrator):
    def __init__(self, model, full_model, calib_loader, metric="hessian_perturb", use_mean_hessian=True, temp=20):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.metric = metric
        self.use_mean_hessian = use_mean_hessian
        self.blocks = {}
        self.full_blocks = {}
        self.raw_pred_softmaxs = None
        self.temperature = temp

        for name, module in self.model.named_modules():
            if len(name.split('.')) >= 2 and name.split('.')[-2] == 'blocks':
                self.blocks[name] = module
                MLPReconstructor._prepare_module_data_init(module)
        for name, module in self.full_model.named_modules():
            if len(name.split('.')) >= 2 and name.split('.')[-2] == 'blocks':
                self.full_blocks[name] = module
                MLPReconstructor._prepare_module_data_init(module)

    @staticmethod
    def _prepare_module_data_init(module):
        module.mlp.fc1.raw_input = module.mlp.fc1.tmp_input = None
        module.mlp.fc2.raw_input = module.mlp.fc2.tmp_input = None
        module.mlp.raw_out = module.mlp.tmp_out = None
        module.mlp.raw_grad = module.mlp.tmp_grad = None
        module.mlp.forward = MethodType(mlp_forward, module.mlp)
        module.mlp.perturb_u = module.mlp.perturb_d = False

    def init_block_raw_data(self, block, device):
        self.init_block_raw_inp_outp(block, device)
        if self.metric == "hessian_perturb":
            self.init_block_perturb_hessian(block, device)

    def init_block_raw_inp_outp(self, block, device):
        hooks = []
        hooks.append(block.mlp.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.mlp.fc1.register_forward_hook(self.single_input_forward_hook))
        hooks.append(block.mlp.fc2.register_forward_hook(self.single_input_forward_hook))
        need_calculate_raw_softmax = False
        if self.raw_pred_softmaxs is None and self.metric == "hessian_perturb":
            need_calculate_raw_softmax = True
            self.raw_pred_softmaxs = []
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.full_model(inp) / self.temperature
                if need_calculate_raw_softmax:
                    raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                    self.raw_pred_softmaxs.append(raw_pred_softmax)
            torch.cuda.empty_cache()
        block.mlp.raw_out = torch.cat(block.mlp.tmp_out, dim=0)
        block.mlp.fc1.raw_input = torch.cat(block.mlp.fc1.tmp_input, dim=0)
        block.mlp.fc2.raw_input = torch.cat(block.mlp.fc2.tmp_input, dim=0)
        block.mlp.fc1.tmp_input = block.mlp.fc2.tmp_input = block.mlp.tmp_out = None
        for hook in hooks:
            hook.remove()

    def init_block_perturb_hessian(self, block, device):
        raw_grads = []
        for step in range(2):
            hook = block.mlp.register_full_backward_hook(self.grad_hook)
            block.mlp.perturb_u, block.mlp.perturb_d = (step == 0, step == 1)
            for i, (inp, target) in enumerate(self.calib_loader):
                self.model.zero_grad()
                inp = inp.to(device)
                pred = self.full_model(inp) / self.temperature
                loss = F.kl_div(F.log_softmax(pred, dim=-1), self.raw_pred_softmaxs[i], reduction="batchmean")
                loss.backward()
            torch.cuda.empty_cache()
            raw_grads.append(torch.cat(block.mlp.tmp_grad, dim=0))
            block.mlp.tmp_grad = None
            block.mlp.perturb_u = block.mlp.perturb_d = False
            hook.remove()
        block.mlp.raw_grad = (raw_grads[0] - raw_grads[1]).abs()
        block.mlp.raw_grad = block.mlp.raw_grad.mean(dim=0, keepdim=True) if self.use_mean_hessian else block.mlp.raw_grad
        block.mlp.raw_grad = block.mlp.raw_grad * torch.sqrt(block.mlp.raw_grad.numel() / block.mlp.raw_grad.pow(2).sum())
            
    def reconstruct_single_block(self, name, block, device, ub,
                                 batch_size: int = 32, iters: int = 20000, lr: float = 4e-5, p: float = 2.0):
        w_params = []
        for _name, module in block.named_modules():
            if 'fc1' in _name or 'fc2' in _name or 'norm2' in _name:
                w_params += [module.weight, module.bias]
        w_optimizer = torch.optim.Adam(w_params, lr=lr)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=iters, eta_min=0.)
        loss_func = LossFunction(block, weight=2.0, rec_loss=self.metric, max_count=iters, p=p)
        for i in range(iters):
            idx = torch.randperm(block.mlp.fc1.raw_input.size(0))[:batch_size]
            cur_inp = block.mlp.fc1.raw_input[idx].to(device)
            cur_out = block.mlp.raw_out[idx].to(device)
            if self.metric == "hessian_perturb":
                cur_grad = block.mlp.raw_grad.to(device) if self.use_mean_hessian else block.mlp.raw_grad[idx].to(device)
            else:
                cur_grad = None
            w_optimizer.zero_grad()
            recon_out = block.mlp(cur_inp)
            fc2_inp = block.mlp.act(block.mlp.fc1(cur_inp))
            fc2_quant_inp = torch.clamp(fc2_inp, 0, ub)
            quant_out = block.mlp.fc2(fc2_quant_inp)
            err = loss_func(recon_out, cur_out, cur_grad, quant_out)
            err.backward()
            w_optimizer.step()
            w_scheduler.step()
        del block.mlp.fc1.raw_input, block.mlp.raw_out, block.mlp.raw_grad
        torch.cuda.empty_cache()

    def reconstruct_model(self, pct):
        device = next(self.model.parameters()).device
        for name, block in self.blocks.items():
            logging.info('reconstructing {} ...'.format(name))
            full_block = self.full_blocks[name]
            self.init_block_raw_data(full_block, device)
            block.mlp.fc1.raw_input = full_block.mlp.fc1.raw_input.to(device)
            block.mlp.raw_out = full_block.mlp.raw_out.to(device)
            if self.metric == "hessian_perturb":
                block.mlp.raw_grad = full_block.mlp.raw_grad.to(device)
                del full_block.mlp.raw_grad
            ub = positive_percentile(full_block.mlp.fc2.raw_input, pct=pct)
            del full_block.mlp.fc1.raw_input, full_block.mlp.fc2.raw_input, full_block.mlp.raw_out
            logging.info('ub: {}'.format(ub))
            self.reconstruct_single_block(name, block, device, ub=ub)
            logging.info('finished reconstructing {}.'.format(name))

        
class LossFunction:
    def __init__(self,
                 block,
                 weight: float = 2.0,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.2,
                 p: float = 2.):
        self.block = block
        self.rec_loss = rec_loss
        self.weight = weight
        self.p = p
        self.count = 0
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt, grad=None, quant_out=None):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=self.p) / 10
            quant_loss = self.lp_loss(quant_out, tgt, p=self.p) / 10
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0) / 10
        elif self.rec_loss == 'hessian_perturb':
            rec_loss = ((pred - tgt).pow(2) * grad.abs()).sum(1).mean() / 10
            quant_loss = ((quant_out - tgt).pow(2) * grad.abs()).sum(1).mean() / 10
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        total_loss = rec_loss + quant_loss * self.weight
        if self.count == 1 or self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, quant:{:.3f})\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(quant_loss), self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
            