import torch
from torch import nn


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x


def ceil_ste(x: torch.Tensor):
    return (x.ceil() - x).detach() + x
