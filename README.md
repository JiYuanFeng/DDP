# 🎆 DDP: Diffusion Model for Dense Visual Prediction

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddp-diffusion-model-for-dense-visual/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=ddp-diffusion-model-for-dense-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddp-diffusion-model-for-dense-visual/monocular-depth-estimation-on-sun-rgbd)](https://paperswithcode.com/sota/monocular-depth-estimation-on-sun-rgbd?p=ddp-diffusion-model-for-dense-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddp-diffusion-model-for-dense-visual/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=ddp-diffusion-model-for-dense-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddp-diffusion-model-for-dense-visual/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=ddp-diffusion-model-for-dense-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ddp-diffusion-model-for-dense-visual/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=ddp-diffusion-model-for-dense-visual)

The official implementation of the paper "[DDP: Diffusion Model for Dense Visual Prediction](https://arxiv.org/abs/2303.17559)".

### [Project Page](https://github.com/JiYuanFeng/DDP) | [Paper](https://arxiv.org/abs/2303.17559)


This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for DDP, which contains:

- [x] Semantic Segmentation
- [x] Depth Estimation
- [x] BEV Map Segmentation
- [x] Mask Conditioned ControlNet

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), [BEVfusion](https://github.com/mit-han-lab/bevfusion), [ControlNet](https://github.com/lllyasviel/ControlNet), as the correspond codebase.
We would like to express our sincere gratitude to the developers of these codebases.

## News
In the coming days, we will be updating the corresponding codebases.


## Abstract

We propose a simple, efficient, yet powerful framework for dense visual predictions based on the conditional diffusion pipeline. Our approach follows a "noise-to-map" generative paradigm for prediction by progressively removing noise from a random Gaussian distribution, guided by the image. The method, called DDP, efficiently extends the denoising diffusion process into the modern perception pipeline. Without task-specific design and architecture customization, DDP is easy to generalize to most dense prediction tasks, e.g., semantic segmentation and depth estimation. In addition, DDP shows attractive properties such as dynamic inference and uncertainty awareness, in contrast to previous single-step discriminative methods. We show top results on three representative tasks with six diverse benchmarks, without tricks, DDP achieves state-of-the-art or competitive performance on each task compared to the specialist counterparts.


## Method
<img width="1680" alt="image" src="https://github.com/JiYuanFeng/DDP/assets/23737120/e45b0241-79c4-4cad-886f-ec1250b61412">



## Usage
please refer to each task folder for more details.

## Catalog
- [ ] Depth Estimation checkpoints
- [x] Depth Estimation code
- [x] BEVMap checkpoints
- [x] BEVMap Segmentation code
- [x] Mask Conditioned ControlNet checkpoints
- [x] Mask Conditioned ControlNet code
- [x] Segmentation checkpoints
- [x] Segmentation code
- [x] Initialization

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{ji2023ddp,
  title={DDP: Diffusion Model for Dense Visual Prediction},
  author={Ji, Yuanfeng and Chen, Zhe and Xie, Enze and Hong, Lanqing and Liu, Xihui and Liu, Zhaoqiang and Lu, Tong and Li, Zhenguo and Luo, Ping},
  journal={arXiv preprint arXiv:2303.17559},
  year={2023}
}
```






