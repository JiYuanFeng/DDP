# Applying DDP for Semantic Segmentation

Our segmentation code is developed on top of MMSegmentation v0.20.2.

For details please see [DDP](https://arxiv.org/abs/2303.17559)

---

If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{ji2023ddp,
  title={DDP: Diffusion Model for Dense Visual Prediction},
  author={Ji, Yuanfeng and Chen, Zhe and Xie, Enze and Hong, Lanqing and Liu, Xihui and Liu, Zhaoqiang and Lu, Tong and Li, Zhenguo and Luo, Ping},
  journal={arXiv preprint arXiv:2303.17559},
  year={2023}
}
```

## Installation
The code is based on the MMSegmentation v0.29.0+

```
# recommended to create a new environment with torch1.12 + cuda11.6
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
cd DDP/segmentation/
pip install -v -e .
```
## Data Preparation

Preparing ADE20K/Cityscapes according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Results and Models

**Cityscapes val (step 3)**

|  Backbone  | Lr schd | Crop Size | mIoU (SS/MS) | #Param |                                   Config                                    |                                                                                                        Download                                                                                                        |
|:----------:|:-------:|:---------:|:------------:|:------:|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ConvNext-T |  160K   | 512x1024  | 82.60/83.15  |  40M   | [config](configs/cityscapes/ddp_convnext_t_4x4_512x1024_160k_cityscapes.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_t_4x4_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_t_4x4_512x1024_160k_cityscapes.log) |
| ConvNext-S |  160K   | 512x1024  | 82.69/83.58  |  62M   | [config](configs/cityscapes/ddp_convnext_s_4x4_512x1024_160k_cityscapes.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_s_4x4_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_s_4x4_512x1024_160k_cityscapes.log) |
| ConvNext-B |  160K   | 512x1024  | 82.78/83.49  |  100M  | [config](configs/cityscapes/ddp_convnext_b_4x4_512x1024_160k_cityscapes.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_b_4x4_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_b_4x4_512x1024_160k_cityscapes.log) |
| ConvNext-L |  160K   | 512x1024  | 83.21/83.92  |  209M  | [config](configs/cityscapes/ddp_convnext_l_4x4_512x1024_160k_cityscapes.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_l_4x4_512x1024_160k_cityscapes.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_l_4x4_512x1024_160k_cityscapes.log) |

[//]: # (|   Swin-T   |  160K   | 512x1024  | 81.24/82.46  |  39M   |   [config]&#40;./configs/ddp/ddp_swin_t_4x4_512x1024_160k_cityscapes.py&#41;   | [ckpt]&#40;&#41; \ [log]&#40;&#41; |)
[//]: # (|   Swin-S   |  160K   | 512x1024  | 82.41/83.21  |  61M   |   [config]&#40;./configs/ddp/ddp_swin_s_4x4_512x1024_160k_cityscapes.py&#41;   | [ckpt]&#40;&#41; \ [log]&#40;&#41; |)
[//]: # (|   Swin-B   |  160K   | 512x1024  | 82.54/83.42  |  99M   |   [config]&#40;./configs/ddp/ddp_swin_b_4x4_512x1024_160k_cityscapes.py&#41;   | [ckpt]&#40;&#41; \ [log]&#40;&#41; |)

**Cityscapes val (with self-aligned denoising, single scale)**

|  Backbone  | Lr schd | Crop Size | mIoU (step 1/3/10)    | #Param |                                   Config                                          |                                                   Download                                                     |
|:----------:|:-------:|:---------:|:---------------------:|:------:|:---------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| ConvNext-T |  5K     | 512x1024  |    82.3/82.6/82.6     |  40M   | [config](configs/cityscapes/ddp_convnext_t_4x4_512x1024_5k_cityscapes_aligned.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_t_4x4_512x1024_5k_cityscapes_aligned.pth)    \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_t_4x4_512x1024_5k_cityscapes_aligned.log)  |
| ConvNext-L |  5K     | 512x1024  |    83.0/83.2/83.2        |  209M  | [config](configs/cityscapes/ddp_convnext_l_4x4_512x1024_5k_cityscapes_aligned.py) | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_l_4x4_512x1024_5k_cityscapes_aligned.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_convnext_l_4x4_512x1024_5k_cityscapes_aligned.log)  |


**ADE20k val (step 3)**

| Backbone | Lr schd | Crop Size | mIoU (SS/MS) | #Param |                            Config                             |                                                                                               Download                                                                                               |
|:--------:|:-------:|:---------:|:------------:|:------:|:-------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Swin-T  |  160K   |  512x512  |  47.0/47.8   |  40M   | [config](configs/ade/ddp_swin_t_2x8_512x512_160k_ade20k.py)   | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_t_2x8_512x512_160k_ade20k.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_t_2x8_512x512_160k_ade20k.log)      |
|  Swin-S  |  160K   |  512x512  |  48.7/49.7   |  61M   | [config](configs/ade/ddp_swin_s_2x8_512x512_160k_ade20k.py)   | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_s_2x8_512x512_160k_ade20k.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_s_2x8_512x512_160k_ade20k.log)      |
|  Swin-B  |  160K   |  512x512  |  49.4/50.8   |  99M   | [config](configs/ade/ddp_swin_b_2x8_512x512_160k_ade20k.py)   | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_b_2x8_512x512_160k_ade20k.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_b_2x8_512x512_160k_ade20k.log)      |
|  Swin-L  |  160K   |  512x512  |  53.2/54.4   |  207M  | [config](configs/ade/ddp_swin_l_2x8_512x512_160k_ade20k.py)   | [ckpt](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_l_2x8_512x512_160k_ade20k.pth) \| [log](https://huggingface.co/yfji/DDP-Weight/resolve/main/ddp_swin_l_2x8_512x512_160k_ade20k.log)      |

## Training

Multi-gpu training
```
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```
For example, To train DDP-ConvNext-L on cityscapes with 4 gpus run:
```
bash tools/dist_train.sh configs/cityscapes/ddp_convnext_l_4x4_512x1024_160k_cityscapes.py 4
```
Self-aligned denoising
```
bash tools/dist_train.sh configs/cityscapes/ddp_convnext_t_4x4_512x1024_5k_cityscapes_aligned.py 4
```

## Evaluation

Single-gpu testing
```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU
```

Multi-gpu testing
```
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval mIoU
```

For example, To evaluate DDP-ConvNext-T on cityscapes val on a single node with 4 gpus run:
```
bash tools/dist_test.sh configs/cityscapes/ddp_convnext_t_4x4_512x1024_160k_cityscapes.py ckpts/ddp_convnext_t_4x4_512x1024_160k_cityscapes.pth 4 --eval mIoU
```
This should give the below results. Note that the results may vary a little on different machines due to the randomness of the diffusion modeling.
```
Summary:

+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 96.85 | 82.71 | 88.76 |
+-------+-------+-------+
```

## Image Demo

To inference a single image like this:
```
python image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${PRED_FILE} --device cuda:0 --palette ${PALETTE_FILE}
```
For example, the result will be saved in `resources/demo_pred.png` by running:
```
python image_demo.py resources/demo.png configs/ddp/ddp_convnext_t_4x4_512x1024_160k_cityscapes.py ckpts/ddp_convnext_t_4x4_512x1024_160k_cityscapes.pth resources/demo_pred.png --device cuda:0 --palette cityscapes
```




