# Applying DDP for Depth Estimation

Our depth estimation code is developed on top of Our segmentation code is developed on top of MMSegmentation v0.20.2.

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
The code is based on the Monocular-Depth-Estimation-Toolbox.

```
# recommended to create a new environment with torch1.12 + cuda11.6
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
cd DDP/depth/
pip install -v -e .
```
## Data Preparation

Preparing Kitti/NYUv2 according to the [guidelines](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/blob/main/docs/dataset_prepare.md#prepare-datasets) in Monocular-Depth-Estimation-Toolbox.

## Results and Models

**Kitti (step 3)**

| Backbone | Lr schd | Crop Size | REL | RMSE |                                   Config                                   |     Download     |
|:--------:|:-------:|:---------:|-----|:----:|:--------------------------------------------------------------------------:|:----------------:|
|  Swin-T  |   38K   |  352x704  | TBA | TBA  |     [config](configs/ddp_kitti/ddp_swint_1k_w7_kitti_bs2x8_scale01.py)     |    TBA \ TBA     |
|  Swin-S  |   38K   |  352x704  | TBA | TBA  |     [config](configs/ddp_kitti/ddp_swins_1k_w7_kitti_bs2x8_scale01.py)     |    TBA \ TBA     |
|  Swin-B  |   38K   |  352x704  | TBA | TBA  |    [config](configs/ddp_kitti/ddp_swinb_22k_w7_kitti_bs2x8_scale01.py)     |    TBA \ TBA     |
|  Swin-L  |   38K   |  352x704  | TBA | TBA  |    [config](configs/ddp_kitti/ddp_swinl_22k_w7_kitti_bs2x8_scale01.py)     |    TBA \ TBA     |

**NYUV2 (step 3)**

| Backbone | Lr schd | Crop Size | REL | RMSE |                                 Config                                  |     Download     |
|:--------:|:-------:|:---------:|-----|:----:|:-----------------------------------------------------------------------:|:----------------:|
|  Swin-T  |   38K   |  416x544  | TBA | TBA  |     [config](configs/ddp_nyu/ddp_swint_1k_w7_nyu_bs2x8_scale01.py)    |    TBA \ TBA     |
|  Swin-S  |   38K   |  416x544  | TBA | TBA  |     [config](configs/ddp_nyu/ddp_swins_1k_w7_nyu_bs2x8_scale01.py)    |    TBA \ TBA     |
|  Swin-B  |   38K   |  416x544  | TBA | TBA  |     [config](configs/ddp_nyu/ddp_swinb_22k_w7_nyu_bs2x8_scale01.py)   |    TBA \ TBA     |
|  Swin-L  |   38K   |  416x544  | TBA | TBA  |     [config](configs/ddp_nyu/ddp_swinl_22k_w12_nyu_bs2x8_scale01.py)  |    TBA \ TBA     |



## Training

Multi-gpu training
```
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```
For example, To train DDP-Swin-T on Kitti with 8 gpus run:
```
bash tools/dist_train.sh configs/ddp_kitti/ddp_swint_1k_w7_kitti_bs2x8_scale01.py 8
```

## Evaluation

Single-gpu testing
```
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval xx
```

Multi-gpu testing
```
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval xx
```


