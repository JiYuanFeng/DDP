# Apply DDP for Semantic Segmentation

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
# recommended to create a new environment with torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
cd DDP/segmentation/
pip install -v -e .
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```
## Data Preparation

Preparing ADE20K/Cityscapes according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Results and Models

**Cityscapes val**

| Method | Backbone |  Pretrain  | Lr schd | Crop Size | mIoU (SS/MS)                                                                                                                                                                            | #Param |                                        Config                                         | Download                                                                                                                                                                                                      |
|:------:|:--------:|:----------:|:-------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

**ADE20k val**

## Training

## Evaluation

## Image Demo



