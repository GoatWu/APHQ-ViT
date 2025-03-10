import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from types import MethodType
import logging
import random
import copy
from tqdm import tqdm

import sys
sys.path.append('..')
from quant_utils.block_recon import BlockReconstructor
from quant_utils.calibrator import QuantCalibrator
from quantizers._ste import *
from quantizers.adaround import AdaRoundQuantizer
from quantizers.uniform import UniformQuantizer
from quant_layers import *

from mmdet.models.backbones.swin_transformer import window_partition, window_reverse


def mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-3
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-3
    return x


def binary_kl(pred, target):
    preds = torch.clamp(pred, min=1e-7, max=1-1e-7)
    targets = torch.clamp(target, min=1e-7, max=1-1e-7)
    return (targets * (torch.log(targets) - torch.log(preds)) + \
       (1 - targets) * (torch.log(1 - targets) - torch.log(1 - preds))).mean()


def positive_percentile(tensor, pct):
    print('calculating fc2 activation percentile ...', flush=True)
    raw_tensor = torch.cat([_.view(-1) for _ in tensor[:128]]) if isinstance(tensor, list) else tensor.view(-1)
    raw_tensor = raw_tensor.cuda()
    mini_batch_size = 1
    tensor_too_large = True
    while tensor_too_large:
        try:
            t = raw_tensor.view(mini_batch_size, -1)[0:1, :]
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
    del raw_tensor
    return result.item()


class MLPReconstructor(BlockReconstructor):
    def __init__(self, model, full_model, calib_loader, calib_size, metric="hessian_perturb", keep_gpu=False):
        QuantCalibrator.__init__(self, model, calib_loader, calib_size)
        self.full_model = full_model
        self.metric = metric
        self.blocks = {}
        self.full_blocks = {}
        self.raw_pred = None
        self.calculate_tag = False

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
        module.mlp.raw_input = module.mlp.tmp_input = None
        module.mlp.fc2.raw_input = module.mlp.fc2.tmp_input = None
        module.mlp.raw_out = module.mlp.tmp_out = None
        module.mlp.raw_grad = module.mlp.tmp_grad = None
        module.mlp.forward = MethodType(mlp_forward, module.mlp)
        module.mlp.perturb_u = module.mlp.perturb_d = False

    def init_block_raw_data(self, block, device):
        self.init_block_raw_inp_outp(block, device)
        if self.metric in ["hessian", "jacobian", "hessian_perturb"]:
            self.init_block_raw_grad(block, device)

    def init_block_raw_inp_outp(self, block, device):
        print('initializing raw input and raw output ...', flush=True)
        hooks = []
        hooks.append(block.mlp.register_forward_hook(self.outp_forward_hook))
        hooks.append(block.mlp.register_forward_hook(self.single_input_forward_hook))
        hooks.append(block.mlp.fc2.register_forward_hook(self.single_input_forward_hook))
        need_calculate_raw_pred = False
        if not self.calculate_tag and self.metric == "hessian_perturb":
            self.calculate_tag = True
            need_calculate_raw_pred = True
            self.rpn_cls, self.rpn_bbox = [], []
            self.full_model.rpn_head.tmp_out = None
            hooks.append(self.full_model.rpn_head.register_forward_hook(self.rpn_outp_forward_hook_detach))
        with torch.no_grad():
            with tqdm(total=self.calib_size) as progress_bar:
                for i, calib_data in enumerate(self.calib_loader):
                    img = [t.to(device) for t in calib_data['img'].data]
                    img_metas = calib_data['img_metas'].data
                    pred = self.full_model(img=img, img_metas=img_metas, return_loss=False)
                    progress_bar.update()
                    if i == self.calib_size - 1:
                        break
            torch.cuda.empty_cache()
        if need_calculate_raw_pred:
            with torch.no_grad():
                self.rpn_cls = self.full_model.rpn_head.tmp_out[0]
                for i in range(len(self.rpn_cls)):
                    for j in range(len(self.rpn_cls[i])):
                        self.rpn_cls[i][j] = torch.sigmoid(self.rpn_cls[i][j].cuda()).cpu()
                self.rpn_bbox = self.full_model.rpn_head.tmp_out[1]
        block.mlp.raw_out = block.mlp.tmp_out
        block.mlp.raw_input = block.mlp.tmp_input
        block.mlp.fc2.raw_input = block.mlp.fc2.tmp_input
        block.mlp.tmp_input = block.mlp.fc2.tmp_input = block.mlp.tmp_out = None
        for hook in hooks:
            hook.remove()

    def init_block_raw_grad(self, block, device):
        print('initializing raw grad ...', flush=True)
        block.tmp_grad = None
        raw_grads = [[], []]
        hooks = []
        hooks.append(self.full_model.rpn_head.register_forward_hook(self.rpn_outp_forward_hook))
        hooks.append(block.mlp.register_full_backward_hook(self.grad_hook))
        for step in range(2):
            block.mlp.perturb_u, block.mlp.perturb_d = (step == 0, step == 1)
            with tqdm(total=self.calib_size) as progress_bar:
                progress_bar.set_description(f"step {step}")
                for i, calib_data in enumerate(self.calib_loader):
                    self.full_model.zero_grad()
                    self.full_model.rpn_head.tmp_out = None
                    img = [t.to(device) for t in calib_data['img'].data]
                    img_metas = calib_data['img_metas'].data
                    _ = self.full_model(img=img, img_metas=img_metas, return_loss=False)                
                    loss_rpn_cls, loss_rpn_bbox = [], []
                    for j in range(len(self.full_model.rpn_head.tmp_out[0])):
                        raw_rpn_cls = self.rpn_cls[i][j].cuda()
                        raw_rpn_bbox = self.rpn_bbox[i][j].cuda()
                        t1 = self.full_model.rpn_head.tmp_out[0][j]
                        t2 = self.full_model.rpn_head.tmp_out[1][j]
                        loss_rpn_cls.append(binary_kl(torch.sigmoid(t1), raw_rpn_cls))
                        mask = (raw_rpn_cls > 0.7).repeat_interleave(4, dim=1).contiguous()
                        loss_rpn_bbox.append(F.smooth_l1_loss(t2 * mask, raw_rpn_bbox * mask, reduction='mean'))
                    loss_rpn_cls = sum(loss_rpn_cls)
                    loss_rpn_bbox = sum(loss_rpn_bbox)
                    loss = loss_rpn_cls / loss_rpn_cls.detach() + loss_rpn_bbox / loss_rpn_bbox.detach()
                    loss.backward()
                    progress_bar.update()
                    if i == self.calib_size - 1:
                        break
            raw_grads[step] = block.mlp.tmp_grad
            block.mlp.tmp_grad = None
            torch.cuda.empty_cache()
            block.perturb_u = block.perturb_d = False
            
        for hook in hooks:
            hook.remove()
        
        block.mlp.raw_grad = []
        with torch.no_grad():
            for i in range(len(raw_grads[0])):
                grad = (raw_grads[0][i].to(device) - raw_grads[1][i].to(device)).abs()
                block.mlp.raw_grad.append((grad * grad.numel() / grad.sum()).cpu())
            
    def reconstruct_single_block(self, name, block, device, ub,
                                 batch_size: int = 32, iters: int = 20000, lr: float = 4e-5, p: float = 2.0):
        w_params = []
        for _name, module in block.named_modules():
            if 'fc1' in _name or 'fc2' in _name:
                w_params += [module.weight, module.bias]
        w_optimizer = torch.optim.Adam(w_params, lr=lr)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=iters, eta_min=0.)
        loss_func = LossFunction(block, weight=2.0, rec_loss=self.metric, max_count=iters, p=p)
        for it in range(iters):
            idx = it % len(block.mlp.raw_out)
            cur_inp = block.mlp.raw_input[idx].to(device)
            cur_out = block.mlp.raw_out[idx].to(device)
            cur_grad = block.mlp.raw_grad[idx].to(device) if self.metric == "hessian_perturb" else None
            w_optimizer.zero_grad()
            recon_out = block.mlp(cur_inp)
            fc2_inp = block.mlp.act(block.mlp.fc1(cur_inp))
            fc2_quant_inp = torch.clamp(fc2_inp, 0, ub)
            quant_out = block.mlp.fc2(fc2_quant_inp)
            err = loss_func(recon_out, cur_out, cur_grad, quant_out)
            err.backward()
            w_optimizer.step()
            w_scheduler.step()
        del block.mlp.raw_input, block.mlp.raw_out, block.mlp.raw_grad
        torch.cuda.empty_cache()

    def reconstruct_model(self, pct):
        device = next(self.model.parameters()).device
        for name, block in self.blocks.items():
            logging.info('reconstructing {} ...'.format(name))
            full_block = self.full_blocks[name]
            self.init_block_raw_data(full_block, device)
            block.mlp.raw_input = full_block.mlp.raw_input
            block.mlp.raw_out = full_block.mlp.raw_out
            if self.metric == "hessian_perturb":
                block.mlp.raw_grad = full_block.mlp.raw_grad
                del full_block.mlp.raw_grad
            ub = positive_percentile(full_block.mlp.fc2.raw_input, pct=pct)
            del full_block.mlp.raw_input, full_block.mlp.fc2.raw_input, full_block.mlp.raw_out
            print('ub: {}'.format(ub))
            self.reconstruct_single_block(name, block, device, ub=ub)
            print('finished reconstructing {}.'.format(name))

        
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
        if self.count == 1 or self.count % 5000 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, quant:{:.3f})\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(quant_loss), self.count), flush=True)
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
            