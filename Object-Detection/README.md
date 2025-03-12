## APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

This repository is adopted from [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) repo.

## Getting Started

- Install pytorch and [MMCV](https://github.com/open-mmlab/mmcv).

```bash
pip install torch==1.10.0 torchvision --index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmcv-full==1.3.17
```

- If your pytorch version is higher than 1.10, you may need to install the bug-fixed version of `mmcv-full==1.3.17` through local compilation instead of openmim:

```bash
git clone https://github.com/GoatWu/mmcv-v1.3.17.git
cd mmcv-v1.3.17
MMCV_WITH_OPS=1 pip install -e . -v
```

- Install `mmpycocotools` using the local installation package, as the version provided by the official source has some bugs.

```bash
cd Object-Detection
pip install mmpycocotools-12.0.3.tar.gz
```

- Install [MMDetection](https://github.com/open-mmlab/mmdetection).

```bash
pip install -v -e .
```

- Download pre-trained models from [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and put them in the `checkpoint` folder.

- link the COCO2017 dataset to the `data/coco` folder.

## Evaluation

You can quantize and evaluate a single model using the following command:

Example: Quantize `Mask R-CNN` with `Swin-T` at W4/A4 precision:

```bash
python tools/test.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py checkpoint/mask_rcnn_swin_tiny_patch4_window7.pth --eval bbox segm --quant-config ./tools/quant_configs/4bit.py
```

## Results

Below are the experimental results of our proposed APHQ-ViT on COCO dataset.

| Model                    | Full Prec.<br>(AP<sup>box</sup> / AP<sup>mask</sup>) | MLP Recon.<br>(AP<sup>box</sup> / AP<sup>mask</sup>) | W4/A4<br>(AP<sup>box</sup> / AP<sup>mask</sup>) |
|:------------------------:|:-----------:|:-----------:|:-----------:|
| Mask-RCNN-Swin-T         | 46.0 / 41.6 | 45.8 / 41.5 | 38.9 / 38.1 |
| Mask-RCNN-Swin-S         | 48.5 / 43.3 | 48.1 / 43.1 | 44.1 / 41.0 |
| Cascade-Mask-RCNN-Swin-T | 50.4 / 43.7 | 50.2 / 43.6 | 48.9 / 42.7 |
| Cascade-Mask-RCNN-Swin-S | 51.9 / 45.0 | 51.7 / 44.7 | 50.3 / 43.7 |
