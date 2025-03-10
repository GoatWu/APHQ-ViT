import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from quant_layers import MinMaxQuantMatMul, MinMaxQuantConv2d, MinMaxQuantLinear


class QuantCalibrator:
    def __init__(self, model, calib_loader):
        self.model = model
        self.calib_loader = calib_loader
        
    def single_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())
        
    def double_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = [[],[]]
        module.tmp_input[0].append(inp[0].cpu().detach())
        module.tmp_input[1].append(inp[1].cpu().detach())
    
    def outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.cpu().detach())
        
    def grad_hook(self, module, grad_input, grad_output):
        if module.tmp_grad is None:
            module.tmp_grad = []
        module.tmp_grad.append(grad_output[0].clone().cpu().detach())

    def batching_quant_calib(self):
        device = next(self.model.parameters()).device
        raw_pred_softmaxs = []
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                pred = self.model(inp)
                raw_pred_softmax = F.softmax(pred, dim=-1).detach()
                raw_pred_softmaxs.append(raw_pred_softmax)
            torch.cuda.empty_cache()

        total = sum(1 for name, module in self.model.named_modules() if hasattr(module, 'metric') and not module.calibrated)
        with tqdm(total=total) as progress_bar:
            for name, module in self.model.named_modules():
                if not hasattr(module, 'metric') or module.calibrated:
                    continue
                progress_bar.set_description(f"calibrating {name}")
                hooks = []
                hooks.append(module.register_forward_hook(self.outp_forward_hook))
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                if isinstance(module, MinMaxQuantMatMul):
                    hooks.append(module.register_forward_hook(self.double_input_forward_hook))
                for i, (inp, target) in enumerate(self.calib_loader):
                    self.model.zero_grad()
                    inp = inp.to(device)
                    pred = self.model(inp)
                torch.cuda.empty_cache()
                # replace cached raw_inputs, raw_outs
                module.raw_out = torch.cat(module.tmp_out, dim=0)
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                for hook in hooks:
                    hook.remove()
                module.tmp_input = module.tmp_out = None
                # run hyperparameter_searching
                with torch.no_grad():
                    module.hyperparameter_searching()
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                    torch.cuda.empty_cache()
                progress_bar.update()
        # end calibration
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"
