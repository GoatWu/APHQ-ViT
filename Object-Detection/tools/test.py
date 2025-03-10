import argparse
import importlib
import os
import sys
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from quant_layers import *
from quant_utils.calibrator import QuantCalibrator
from quant_utils.block_recon import BlockReconstructor
from quant_utils.mlp_recon import MLPReconstructor
from quant_utils.wrap_net import wrap_modules_in_net
import copy


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # personal arguments
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument('--quant-config', type=str, default='./tools/quant_configs/4bit.py')
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
        
    # personal configs
    if args.device.startswith('cuda:') and os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print('reset gpu_id={} and restarting the process ...'.format(gpu_id))
        # os.execv(sys.executable, ['python'] + sys.argv)
        os.execv(sys.executable, [sys.executable] + sys.argv)
        
    device = torch.device('cuda:0')
    
    dir_path = os.path.dirname(os.path.abspath(args.quant_config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.quant_config))[0]
    imported_module = importlib.import_module(module_name)
    quant_config = getattr(imported_module, 'Config')
    quant_cfg = quant_config()
    quant_cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else quant_cfg.w_bit
    quant_cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else quant_cfg.a_bit
    for name, value in vars(quant_cfg).items():
        print(f"{name}: {value}")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    print(cfg.data.train)
    # build the dataloader
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)
    test_data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    calib_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=quant_cfg.calib_batch_size, 
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    optim_data_loader = build_dataloader(
        train_dataset,
        # test_dataset,
        samples_per_gpu=1, 
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = test_dataset.CLASSES

    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()
    
    if quant_cfg.reconstruct_mlp:
        print("start {} guided mlp reconstruction".format(quant_cfg.optim_metric))
        for name, module in model.named_modules():
            if name.split('.')[-1] == 'mlp':
                module.act = nn.ReLU()
        model.eval()
        model.to(device)
        mlp_reconstructor = MLPReconstructor(model, full_model, optim_data_loader, quant_cfg.optim_size, metric=quant_cfg.optim_metric)
        mlp_reconstructor.reconstruct_model(pct=quant_cfg.pct)
        print("{} guided mlp reconstruction finished.".format(quant_cfg.optim_metric))
        recon_model = MMDataParallel(model, device_ids=[0])
        recon_model.eval()
        recon_outputs = single_gpu_test(recon_model, test_data_loader, args.show, args.show_dir,
                                        args.show_score_thr)
        print_out(args, cfg, test_dataset, recon_outputs)
        del recon_model
        torch.cuda.empty_cache()
        
    wrap_modules_in_net(model, quant_cfg, reparam=True, recon=quant_cfg.reconstruct_mlp)
    # print(model)
    model.eval()
    model.to(device)
    
    # for _name, _module in model.named_modules():
    #     print(_name, type(_module))
        
    print("start {} guided calibration".format(quant_cfg.calib_metric))
    calib_batch_num = quant_cfg.calib_size // quant_cfg.calib_batch_size
    quant_calibrator = QuantCalibrator(model, calib_data_loader, calib_batch_num)
    quant_calibrator.batching_quant_calib()
    print("{} guided calibration finished.".format(quant_cfg.calib_metric))

    calib_model = MMDataParallel(model, device_ids=[0])
    calib_model.eval()
    calib_outputs = single_gpu_test(calib_model, test_data_loader, args.show, args.show_dir,
                                    args.show_score_thr)
    print_out(args, cfg, test_dataset, calib_outputs)
    del calib_model
    torch.cuda.empty_cache()

    block_reconstructor = BlockReconstructor(model, full_model, optim_data_loader, quant_cfg.optim_size, metric=quant_cfg.optim_metric, keep_gpu=quant_cfg.keep_gpu)
    block_reconstructor.wrap_quantizers_in_net()
    block_reconstructor.reconstruct_model(quant_act=True, mode=quant_cfg.optim_mode, drop_prob=quant_cfg.drop_prob, quant_ratio=quant_cfg.quant_ratio)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    outputs = single_gpu_test(model, test_data_loader, args.show, args.show_dir,
                              args.show_score_thr)
    print_out(args, cfg, test_dataset, outputs)
        
    
def print_out(args, cfg, test_dataset, outputs):
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            test_dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(test_dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
