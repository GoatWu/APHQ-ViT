import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models.backbones.swin_transformer import window_partition, window_reverse


def swin_block_forward(self, x, mask_matrix, H, W):
    B, L, C = x.shape
    # H, W = self.H, self.W
    assert L == H * W, "input feature has wrong size"
    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)
    # pad feature maps to multiples of window size
    pad_l = pad_t = 0
    pad_r = (self.window_size - W % self.window_size) % self.window_size
    pad_b = (self.window_size - H % self.window_size) % self.window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    _, Hp, Wp, _ = x.shape
    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        attn_mask = mask_matrix
    else:
        shifted_x = x
        attn_mask = None
    # partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()
    x = x.view(B, H * W, C)
    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-3
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-3
    return x


def swin_patchmerging_forward(self, x, H, W):
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    x = x.view(B, H, W, C)
    # padding
    pad_input = (H % 2 == 1) or (W % 2 == 1)
    if pad_input:
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
    x = self.norm(x)
    x = self.reduction(x)
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-3
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-3
    return x


def patch_embed_forward(self, x):
    """Forward function."""
    # padding
    _, _, H, W = x.size()
    if W % self.patch_size[1] != 0:
        x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
    if H % self.patch_size[0] != 0:
        x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

    x = self.proj(x)  # B C Wh Ww
    if self.norm is not None:
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
    if self.perturb_u:
        x = x + torch.ones_like(x) * 1e-3
    elif self.perturb_d:
        x = x - torch.ones_like(x) * 1e-3
    return x
