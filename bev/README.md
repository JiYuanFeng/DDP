# Applying DDP for BEV map Segmentation

Our bev segmentation code is developed on top of [BEVFusion](https://github.com/mit-han-lab/bevfusion).

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
The code is based on the BEVFusion, In addition to the necessary Pytorch environment, you need to install the following dependencies:

```
pip3 uninstall -y mmcv
pip3 install timm==0.4.12 thop
pip3 install mmdet==2.20.0
pip3 install addict
pip3 install yapf
pip3 install torchpack
pip3 install nuscenes-devkit
cd DDP/bev/
rm -rf build/
python3 setup.py develop
```
## Data Preparation

Preparing nuScenes according to the [guidelines](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/nuscenes_det.md) in mmdetection3d.
After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```
Please modify the data root in ``bev/configs/nuscenes/default.yaml`` accordingly.
## Results and Models

**NuScenes val set (step 3)**

|    Model     | mIoU |                                      Config                                       |                                                                                                                                Download                                                                                                                                |
|:------------:|:----:|:---------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    Camera    | 59.4 |  [config](configs/nuscenes/seg/ddp-camera-bev256d2-lss-scale001-d5-lr5e-5.yaml)   |                            [ckpt](https://huggingface.co/yfji/DDP-Weight/blob/main/ddp-camera-bev256d2-lss-scale001-d5-lr5e-5.pth) \ [log](https://huggingface.co/yfji/DDP-Weight/blob/main/ddp-camera-bev256d2-lss-scale001-d5-lr5e-5.log)                            |
| Camera+Lidar | 70.6 |  [config](configs/nuscenes/seg/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.yaml)   |                            [ckpt](https://huggingface.co/yfji/DDP-Weight/blob/main/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.pth) \ [log](https://huggingface.co/yfji/DDP-Weight/blob/main/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.log)                            |


## Training

Multi-gpu training
```
torchpack dist-run -np 8 python tools/train.py ${CONFIG_FILE}
```
For example, To train DDP-fusion on NuScenes with 8 gpus run:
```
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/seg/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```


## Evaluation


```
torchpack dist-run -np 8 python tools/test.py configs/nuscenes/seg/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.yaml https://huggingface.co/yfji/DDP-Weight/blob/main/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.pth --eval map
```

## Visualization

```
torchpack dist-run -np 1 python3 tools/visualize.py \
 configs/nuscenes/seg/ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.yaml \
 --checkpoint ddp-fusion-bev256d2-lss-scale001-d5-lr5e-5.pth \
 --out-dir figrures/temp \
 --mode pred
```




