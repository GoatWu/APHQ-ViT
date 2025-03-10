import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm

import sys
sys.path.append('..')
from quant_layers import MinMaxQuantMatMul, MinMaxQuantLinear, MinMaxQuantConv2d
from quant_utils.calibrator import QuantCalibrator
from quant_utils.block_forward import *
from quantizers.adaround import AdaRoundQuantizer

from types import MethodType
from mmdet.models.backbones.swin_transformer import SwinTransformerBlock, PatchMerging, PatchEmbed
from mmdet.models.backbones.swin_transformer import window_partition, window_reverse
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead


def binary_kl(pred, target):
    preds = torch.clamp(pred, min=1e-7, max=1-1e-7)
    targets = torch.clamp(target, min=1e-7, max=1-1e-7)
    return (targets * (torch.log(targets) - torch.log(preds)) + \
       (1 - targets) * (torch.log(1 - targets) - torch.log(1 - preds))).mean()


def to_device_recursive(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device_recursive(item, device) for item in data)
    else:
        return data
    

def mix_tensors_or_lists(input1, input2, p):
    if isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
        mask = torch.rand_like(input1, dtype=torch.float) < p
        return torch.where(mask, input1, input2)
    elif isinstance(input1, list) and isinstance(input2, list):
        mixed = []
        for tensor1, tensor2 in zip(input1, input2):
            mask = torch.rand_like(tensor1, dtype=torch.float) < p
            mixed.append(torch.where(mask, tensor1, tensor2))
        return mixed


class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, full_model, calib_loader, calib_size, metric="hessian_perturb", keep_gpu=False):
        super().__init__(model, calib_loader, calib_size)
        self.full_model = full_model
        self.metric = metric
        self.keep_gpu = keep_gpu
        self.calculate_tag = False
        self.blocks = {}
        self.full_blocks = {}
        self.quanted_blocks = []
        self.raw_pred = None
        types_of_block = [
            PatchEmbed,
            SwinTransformerBlock,
            PatchMerging,
            FPN,
            # RPNHead,
            # ConvFCBBoxHead,
        ]
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in types_of_block):
                self.blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)
        for name, module in self.full_model.named_modules():
            if any(isinstance(module, t) for t in types_of_block):
                self.full_blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)
                
    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
        module.raw_grad = module.tmp_grad = None
        module.quanted_input = module.quanted_out = module.quanted_grad = None
        module.quanted_pred_softmaxs = None
        if isinstance(module, SwinTransformerBlock):
            module.forward = MethodType(swin_block_forward, module)
        elif isinstance(module, PatchMerging):
            module.forward = MethodType(swin_patchmerging_forward, module)
        elif isinstance(module, PatchEmbed):
            module.forward = MethodType(patch_embed_forward, module)
        module.perturb_u = module.perturb_d = False

    def patchmerging_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1])
        module.tmp_input[2].append(inp[2])
        
    def block_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[],[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())
        module.tmp_input[2].append(inp[2])
        module.tmp_input[3].append(inp[3])
        
    def patchembedding_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],]
        module.tmp_input[0].append(inp[0].cpu().detach())
        
    def rpn_outp_forward_hook(self, module, inp, outp):
        module.tmp_out = []
        module.tmp_out.append([t.clone() for t in outp[0]])
        module.tmp_out.append([t.clone() for t in outp[1]])
        
    def rpn_outp_forward_hook_detach(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = [[],[]]
        module.tmp_out[0].append([t.cpu().detach() for t in outp[0]])
        module.tmp_out[1].append([t.cpu().detach() for t in outp[1]])
        
    def roi_outp_forward_hook(self, module, inp, outp):
        module.tmp_out = [outp[0].clone(), outp[1].clone()]
        
    def roi_outp_forward_hook_detach(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = [[],[]]
        module.tmp_out[0].append(outp[0].cpu().detach())
        module.tmp_out[1].append(outp[1].cpu().detach())
    
    def fpn_inp_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append([t.cpu().detach() for t in inp[0]])
    
    def fpn_outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append([t.cpu().detach() for t in outp])
        
    def set_block_mode(self, block, mode='raw'):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode

    def replace_block(self, target_block, new_block):
        self._replace_block_recursive(self.model, target_block, new_block)

    def _replace_block_recursive(self, model, target_block, new_block):
        for name, child in model.named_children():
            if child is target_block:
                setattr(model, name, new_block)
            else:
                self._replace_block_recursive(child, target_block, new_block)
                
    def wrap_quantizers_in_net(self):
        print('wraping quantizers in model ...')
        for name, module in self.model.named_modules():
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.n_V, module.crb_rows, module.in_features), 
                                                           round_mode='learned_hard_sigmoid')
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.weight.shape[0], -1), 
                                                           round_mode='learned_hard_sigmoid')
                module.w_quantizer.soft_targets = True

    def set_qdrop(self, block, prob):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) and hasattr(module.a_quantizer, 'drop_prob'):
                    module.a_quantizer.drop_prob = prob
                elif isinstance(module, MinMaxQuantMatMul):
                    if hasattr(module.A_quantizer, 'drop_prob'):
                        module.A_quantizer.drop_prob = prob
                    if hasattr(module.B_quantizer, 'drop_prob'):
                        module.B_quantizer.drop_prob = prob

    def init_block_raw_data(self, block, full_block, name, device, qinp=False, keep_gpu=False):
        self.init_block_raw_inp_outp(block, full_block, name, device)
        if qinp:
            self.init_block_quanted_input(block, full_block, name, device)
        
        if self.metric == "hessian_perturb" and 'backbone' in name:
            self.init_block_raw_grad(block, full_block, name, device)

        if keep_gpu:
            block.raw_input = to_device_recursive(block.raw_input, device)
            block.raw_out = to_device_recursive(block.raw_out, device)
            if block.quanted_input is not None:
                block.quanted_input = to_device_recursive(block.quanted_input, device)
            if block.raw_grad is not None:
                block.raw_grad = to_device_recursive(block.raw_grad, device)

    def init_block_raw_inp_outp(self, block, full_block, name, device):
        print('initializing raw input and raw output ...', flush=True)    
        full_block.tmp_input, full_block.tmp_out = None, None
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        hooks = []
        if isinstance(block, FPN):
            hooks.append(full_block.register_forward_hook(self.fpn_outp_forward_hook))
            hooks.append(full_block.register_forward_hook(self.fpn_inp_forward_hook))
        elif isinstance(block, RPNHead):
            hooks.append(full_block.register_forward_hook(self.fpn_inp_forward_hook))
        elif isinstance(block, ConvFCBBoxHead):
            hooks.append(full_block.register_forward_hook(self.single_input_forward_hook))
            hooks.append(full_block.register_forward_hook(self.roi_outp_forward_hook_detach))
        else:
            hooks.append(full_block.register_forward_hook(self.outp_forward_hook))
            if isinstance(block, SwinTransformerBlock):
                hooks.append(full_block.register_forward_hook(self.block_forward_hook))
            elif isinstance(block, PatchMerging):
                hooks.append(full_block.register_forward_hook(self.patchmerging_forward_hook))
            elif isinstance(block, PatchEmbed):
                hooks.append(full_block.register_forward_hook(self.patchembedding_forward_hook))
            else:
                raise NotImplementedError
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

        if isinstance(block, RPNHead):
            block.raw_out = [[], []]
            block.raw_out[0] = self.rpn_cls
            block.raw_out[1] = self.rpn_bbox
        elif isinstance(block, ConvFCBBoxHead):
            block.raw_out = full_block.tmp_out
            with torch.no_grad():
                for i in range(len(block.raw_out[0])):
                    block.raw_out[0][i] = torch.softmax(block.raw_out[0][i].cuda(), dim=-1).cpu()
        else:
            block.raw_out = full_block.tmp_out
        block.raw_input = full_block.tmp_input
        del full_block.tmp_input, full_block.tmp_out
        full_block.tmp_input, full_block.tmp_out = None, None
        for hook in hooks:
            hook.remove() 

    def init_block_quanted_input(self, block, full_block, name, device):
        print('initializing quanted input ...', flush=True)
        full_block.tmp_out = None
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'quant_forward' if _name in self.quanted_blocks else 'raw')
        self.replace_block(block, full_block)
        if isinstance(block, (FPN, RPNHead)):
            hook = full_block.register_forward_hook(self.fpn_inp_forward_hook)
        else:
            hook = full_block.register_forward_hook(self.single_input_forward_hook)
        with torch.no_grad():
            with tqdm(total=self.calib_size) as progress_bar:
                for i, calib_data in enumerate(self.calib_loader):
                    img = [t.to(device) for t in calib_data['img'].data]
                    img_metas = calib_data['img_metas'].data
                    pred = self.model(img=img, img_metas=img_metas, return_loss=False)
                    progress_bar.update()
                    if i == self.calib_size - 1:
                        break
        torch.cuda.empty_cache()
        block.quanted_input = full_block.tmp_input
        del full_block.tmp_input
        full_block.tmp_input = None
        hook.remove()
        self.replace_block(full_block, block)
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
    
    def init_block_raw_grad(self, block, full_block, name, device):
        print('initializing raw grad ...', flush=True)
        full_block.tmp_grad = None
        raw_grads = [[], []]
        hooks = []
        hooks.append(self.full_model.rpn_head.register_forward_hook(self.rpn_outp_forward_hook))
        hooks.append(full_block.register_full_backward_hook(self.grad_hook))
        for step in range(2):
            full_block.perturb_u, full_block.perturb_d = (step == 0, step == 1)
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
            raw_grads[step] = full_block.tmp_grad
            full_block.tmp_grad = None
            torch.cuda.empty_cache()
            full_block.perturb_u = full_block.perturb_d = False
            
        for hook in hooks:
            hook.remove()
        
        block.raw_grad = []
        with torch.no_grad():
            for i in range(len(raw_grads[0])):
                grad = (raw_grads[0][i].to(device) - raw_grads[1][i].to(device)).abs()
                block.raw_grad.append((grad * grad.numel() / grad.sum()).cpu())
            
    def reconstruct_single_block(self, name, block, device,
                                 batch_size: int = 1, iters: int = 20000, weight: float = 0.01,
                                 b_range: tuple = (20, 2), warmup: float = 0.2, lr: float = 4e-5, p: float = 2.0, 
                                 quant_act = False, mode = 'qdrop', drop_prob: float = 1.0, quant_ratio = 1.0):
        self.set_block_mode(block, 'quant_forward')
        for _name, module in block.named_modules():
            if hasattr(module, 'a_quantizer') and hasattr(module.a_quantizer, 'training_mode'):
                module.a_quantizer.init_training()
        self.set_qdrop(block, drop_prob)
        w_params, a_params = [], []
        for _name, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        module.a_quantizer.scale.requires_grad = True
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.mode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        module.A_quantizer.scale.requires_grad = True
                        module.B_quantizer.scale.requires_grad = True
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.mode = 'raw'
        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=lr) if len(a_params) != 0 else None
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=iters, eta_min=0.) if len(a_params) != 0 else None
        if 'backbone' in name:
            rec_loss = self.metric
        elif isinstance(block, FPN):
            rec_loss = 'mse'
        elif isinstance(block, RPNHead):
            rec_loss = 'rpn'
        elif isinstance(block, ConvFCBBoxHead):
            rec_loss = 'roi'
        else:
            raise NotImplementedError
        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters, 
                                 rec_loss=rec_loss, b_range=b_range, decay_start=0, warmup=warmup, p=p)

        for it in range(iters):
            idx = it % len(block.raw_out)
            if isinstance(block, (FPN, RPNHead, ConvFCBBoxHead)):
                cur_fp_inp = to_device_recursive(block.raw_input[idx], device)
            else:
                cur_fp_inp = to_device_recursive(block.raw_input[0][idx], device)
            if mode == 'qdrop':
                cur_quant_inp = to_device_recursive(block.quanted_input[idx], device)
                cur_inp = mix_tensors_or_lists(cur_quant_inp, cur_fp_inp, quant_ratio)
            elif mode == 'rinp':
                cur_inp = cur_fp_inp
            elif mode == 'qinp':
                cur_inp = to_device_recursive(block.quanted_input[idx], device)
            if isinstance(block, SwinTransformerBlock):
                cur_inp_1 = to_device_recursive(block.raw_input[1][idx], device)
            if isinstance(block, (RPNHead, ConvFCBBoxHead)):
                cur_out = [[], []]
                cur_out[0] = to_device_recursive(block.raw_out[0][idx], device)
                cur_out[1] = to_device_recursive(block.raw_out[1][idx], device)
            else:
                cur_out = to_device_recursive(block.raw_out[idx], device)
            cur_grad = to_device_recursive(block.raw_grad[idx], device) if block.raw_grad is not None else None
            w_optimizer.zero_grad()
            if quant_act:
                a_optimizer.zero_grad()
            if isinstance(block, SwinTransformerBlock):
                out_quant = block(cur_inp, cur_inp_1, block.raw_input[2][idx], block.raw_input[3][idx])
            elif isinstance(block, PatchMerging):
                out_quant = block(cur_inp, block.raw_input[1][idx], block.raw_input[2][idx])
            elif isinstance(block, (PatchEmbed, ConvFCBBoxHead)):
                out_quant = block(cur_inp)
            elif isinstance(block, FPN):
                out_quant = block(cur_inp)
                out_quant = torch.cat([t.view(-1) for t in out_quant])
                cur_out = torch.cat([t.view(-1) for t in cur_out])
            elif isinstance(block, RPNHead):
                out_quant = block(tuple(cur_inp))
            else:
                raise NotImplementedError
            err = loss_func(out_quant, cur_out, cur_grad)
            err.backward()
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
        torch.cuda.empty_cache()
        # Finish optimization, use hard rounding.
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'a_quantizer') and hasattr(module.a_quantizer, 'training_mode'):
                module.a_quantizer.end_training()
        self.set_qdrop(block, 1.0)
        del block.raw_input, block.raw_out, block.raw_grad, block.quanted_input, block.quanted_out, block.quanted_grad
        torch.cuda.empty_cache()

    def reconstruct_model(self, quant_act: bool = False, mode: str = 'qdrop+', drop_prob: float = 1.0, quant_ratio: float = 1.0):
        device = next(self.model.parameters()).device
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'raw'
        for idx, name in enumerate(self.blocks.keys()):
            block, full_block = self.blocks[name], self.full_blocks[name]
            print('reconstructing {} ...'.format(name))
            self.init_block_raw_data(block, full_block, name, device, qinp=(mode != 'rinp'), keep_gpu=self.keep_gpu)
            self.reconstruct_single_block(name, block, device, quant_act=quant_act, mode=mode, drop_prob=drop_prob, quant_ratio=quant_ratio)
            self.quanted_blocks.append(name)
            print('finished reconstructing {}.'.format(name))
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'quant_forward'
            if hasattr(module, 'w_quantizer') and isinstance(module.w_quantizer, AdaRoundQuantizer):
                module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
                del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"

        
class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.init_loss = 0
        self.init_cls_loss = 0
        self.init_reg_loss = 0
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'mae':
            rec_loss = self.lp_loss(pred, tgt, p=1.0)
        elif self.rec_loss == 'hessian_perturb':
            rec_loss = ((pred - tgt).pow(2) * grad.abs()).sum()
        elif self.rec_loss in ['rpn', 'roi']:
            if self.rec_loss == 'rpn':
                cls_loss, reg_loss = [], []
                for i in range(len(pred[0])):
                    cls_loss.append(binary_kl(torch.sigmoid(pred[0][i]), tgt[0][i]))
                    mask = (tgt[0][i] > 0.7).unsqueeze(2).repeat_interleave(4, dim=2).contiguous().view_as(tgt[1][i])
                    reg_loss.append(F.smooth_l1_loss(pred[1][i] * mask, tgt[1][i] * mask, reduction='mean'))
                cls_loss, reg_loss = sum(cls_loss), sum(reg_loss)
            else:
                cls_loss = F.kl_div(F.log_softmax(pred[0], dim=-1), tgt[0], reduction="batchmean")
                mask = (tgt[0][..., :-1] > 0.7).repeat_interleave(4, dim=-1).contiguous()
                reg_loss = F.smooth_l1_loss(pred[1] * mask, tgt[1] * mask, reduction='mean')
                # reg_loss = torch.tensor([1]).cuda()
            if self.count == 1:
                self.init_cls_loss = cls_loss.item()
                self.init_reg_loss = reg_loss.item()
            rec_loss = cls_loss / self.init_cls_loss + reg_loss / self.init_reg_loss
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))
        if self.count == 1:
            self.init_loss = rec_loss.item()
        rec_loss = rec_loss * 2 / self.init_loss

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = round_loss_pow2 = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if hasattr(module, 'w_quantizer') and isinstance(module.w_quantizer, AdaRoundQuantizer):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count == 1 or self.count % 5000 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count), flush=True)
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
