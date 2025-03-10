import torch
from torch import nn
from quant_layers.linear import *
from quant_layers.matmul import *
from quant_layers.conv import *
from functools import partial
import timm
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from types import MethodType
from tqdm import tqdm


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def vit_attn_forward(self, x):
    B, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q, k = self.q_norm(q), self.k_norm(k)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def swin_attn_forward(self, x, mask=None):
    B_, N, C = x.shape
    x = self.qkv(x)
    qkv = x.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn + self._get_rel_pos_bias()
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
    
    
def wrap_modules_in_net(model, cfg, reparam=False, recon=False):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attn_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(swin_attn_forward, module)

    module_dict={}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, nn.Conv2d):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantConv2d(
                in_channels = module.in_channels, 
                out_channels = module.out_channels,
                kernel_size = module.kernel_size,
                stride = module.stride,
                mode = 'raw',
                w_bit = cfg.w_bit,
                a_bit = cfg.qconv_a_bit,
                metric = cfg.calib_metric,
                calib_batch_size = cfg.calib_batch_size,
                search_round = cfg.search_round,
                eq_n = cfg.eq_n,
            )
            new_module.weight.data.copy_(module.weight.data)
            new_module.bias.data.copy_(module.bias.data)
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, MatMul):
            idx = idx + 1 if idx != 0 else idx
            new_module = AsymmetricallyBatchingQuantMatMul(
                A_bit = cfg.a_bit,
                B_bit = cfg.a_bit,
                mode = 'raw',
                metric = cfg.calib_metric,
                calib_batch_size = cfg.calib_batch_size,
                search_round = cfg.search_round,
                eq_n = cfg.eq_n,
                head_channel_wise = cfg.matmul_head_channel_wise,
                token_channel_wise = cfg.token_channel_wise,
                num_heads = father_module.num_heads,
            )
            setattr(father_module, name[idx:], new_module)
        if isinstance(module, nn.Linear):
            cur_a_bit = cfg.qhead_a_bit if 'head' in name else cfg.a_bit
            linear_kwargs = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'mode': 'raw',
                'w_bit': cfg.w_bit,
                'a_bit': cur_a_bit,
                'metric': cfg.calib_metric,
                'calib_batch_size': cfg.calib_batch_size,
                'search_round': cfg.search_round,
                'eq_n': cfg.eq_n,
                'n_V': 3 if 'qkv' in name else 1,
                'token_channel_wise': cfg.token_channel_wise,
            }
            idx = idx + 1 if idx != 0 else idx
            if cur_a_bit == cfg.w_bit and reparam and ('qkv' in name or 'reduction' in name or 'fc1' in name):
                idxx = father_name.rfind('.')
                idxx = 0 if idxx == -1 else idxx
                grandfather_name = father_name[:idxx]
                if grandfather_name in module_dict:
                    grandfather_module = module_dict[grandfather_name]
                new_module = AsymmetricallyChannelWiseBatchingQuantLinear(
                    **linear_kwargs, 
                )
                if 'qkv' in name:
                    new_module.prev_layer = grandfather_module.norm1
                if 'fc1' in name:
                    new_module.prev_layer = grandfather_module.norm2
                if 'reduction' in name:
                    new_module.prev_layer = father_module.norm
            else: 
                new_module = AsymmetricallyBatchingQuantLinear(
                    **linear_kwargs,
                    post_relu = ('fc2' in name and recon),
                )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            setattr(father_module, name[idx:], new_module)
    return model


def wrap_reparamed_modules_in_net(model):
    module_dict = {}
    for name, module in model.named_modules():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, AsymmetricallyChannelWiseBatchingQuantLinear):
            idx = idx + 1 if idx != 0 else idx
            linear_kwargs = {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
                'mode': module.mode,
                'w_bit': module.w_quantizer.n_bits,
                'a_bit': module.a_quantizer.n_bits,
                'metric': module.metric,
                'calib_batch_size': module.calib_batch_size,
                'search_round': module.search_round,
                'eq_n': module.eq_n,
                'n_V': module.n_V,
                'token_channel_wise': module.token_channel_wise,
            }
            new_module = AsymmetricallyBatchingQuantLinear(**linear_kwargs)
            if (new_module.a_quantizer.scale.shape != module.a_quantizer.scale.shape):
                new_module.a_quantizer.scale.data = module.a_quantizer.scale.data.clone()
            new_module.load_state_dict(module.state_dict())
            new_module.calibrated = True
            new_module.a_quantizer.inited = True
            new_module.w_quantizer.inited = True
            setattr(father_module, name[idx:], new_module)
    return model
    