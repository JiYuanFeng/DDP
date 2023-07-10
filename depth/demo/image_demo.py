# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from depth.apis import inference_depther, init_depther
from depth.utils import colorize


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file', default='demo/demo.jpg')
    parser.add_argument('config', help='Config file', default='work_dirs/diffdepth_swint_1k_w7_kitti_bs2x1/diffdepth_swint_1k_w7_kitti_bs2x1.py')
    parser.add_argument('checkpoint', help='Checkpoint file', default="work_dirs/diffdepth_swint_1k_w7_kitti_bs2x1/best_abs_rel_iter_8000.pth")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_depther(args.config, args.checkpoint, device=args.device)
    # test a single image, the result has a shape of [1, H, W]
    result = inference_depther(model, args.img)
    # color it and get the colored_results [H, W, 3]
    colored_results = colorize(result, vmin=model.decode_head.min_depth, vmax=model.decode_head.max_depth)[0]

    # BGR to RBG and show the results use matplotlib
    # plt.figure()
    # plt.imshow(colored_results[..., ::-1])
    # plt.show()



if __name__ == '__main__':
    main()
