# Applying DDP for Mask Conditioned ControlNet

Our segmentation code is developed on top of ControlNet.

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
```
conda env create -f environment.yaml
conda activate control
pip install -U openmim
mim install mmcv-full==1.6.2.3
cd controlnet/annotator/ddp
python setup.py develop
```
## Model Preparation
1. Download the pretrained model from [here](https://huggingface.co/yfji/DDP-Weight/blob/main/ddp_swin_t_2x8_512x512_160k_ade20k.pth) and put it under `controlnet/annotator/ckpts/ddp_swin_t_2x8_512x512_160k_ade20k`
2. Download the controlnet pretrained model from [here](https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_seg.pth) and put it under `controlnet/models`
3. Launch the gradio server
```
 python gradio_seg2image_ddp.py
```

## Results

![](github_page/ddp.png)



