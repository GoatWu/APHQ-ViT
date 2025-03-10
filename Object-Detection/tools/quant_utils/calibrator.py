import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append('..')
from quant_layers import MinMaxQuantMatMul, MinMaxQuantLinear, MinMaxQuantConv2d


class QuantCalibrator:
    def __init__(self, model, calib_loader, calib_size):
        self.model = model
        self.calib_loader = calib_loader
        self.calib_size = calib_size
        
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
                for i, calib_data in enumerate(self.calib_loader):
                    with torch.no_grad():
                        # data = torch.cat(calib_data['img'].data)
                        # pred = self.model(data.to(device))
                        img = [t.to(device) for t in calib_data['img'].data]
                        img_metas = calib_data['img_metas'].data
                        pred = self.model(img=img, img_metas=img_metas, return_loss=False)
                    break
                    
                if isinstance(module, MinMaxQuantLinear):
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                    module.raw_out = torch.cat(module.tmp_out, dim=0)
                if isinstance(module, MinMaxQuantConv2d):
                    # In RPN, some conv layers might receive inputs of different scales multiple times, 
                    # resulting in the inability to perform concatenation.
                    # pass
                    module.raw_input = module.tmp_input
                    module.raw_out = module.tmp_out
                    
                if isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [torch.cat(_, dim=0) for _ in module.tmp_input]
                    module.raw_out = torch.cat(module.tmp_out, dim=0)
                for hook in hooks:
                    hook.remove()

                # run hyperparameter_searching
                with torch.no_grad():
                    module.hyperparameter_searching()
                    if hasattr(module, 'prev_layer') and module.prev_layer is not None:
                        progress_bar.set_description(f"reparaming {name}")
                        module.reparam()
                    torch.cuda.empty_cache()
                module.mode = "raw"
                progress_bar.update()
        # end calibration
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"
