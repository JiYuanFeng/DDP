import argparse

import numpy as np
import random
import os
import mmcv
from multiprocessing import Process, Pool

import torch
import tqdm
from functools import partial

random.seed(8)  # for reproducibility
np.random.seed(8)

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
               'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']


def perturb(i, p, s):
    img = corrupt(i, corruption_name=p, severity=s)
    return img


def convert_img_path(ori_path, suffix):
    new_path = ori_path.replace('val', suffix)
    assert new_path != ori_path
    return new_path


def count_label(img_path):
    img = mmcv.imread(img_path)
    values, counts = np.unique(img, return_counts=True)
    return {k: v for k, v in zip(values, counts)}


def main(args):
    img_dir = "/data/share_data/cityscapes/gtFine/train"
    data_infos = [os.path.join(img_dir, img_path) for img_path in
                  mmcv.scandir(img_dir, suffix='labelTrainIds.png', recursive=True)]
    # data_infos = data_infos[:10]
    pool = Pool(args.worker)
    stats = list(tqdm.tqdm(pool.imap(partial(count_label), data_infos), total=len(data_infos)))
    import collections
    from collections import Counter
    c = Counter()
    for d in stats:
        c.update(d)
    c.pop(255)
    od = collections.OrderedDict(sorted(c.items()))
    counts = list(od.values())
    print(counts)
    weights = 1 / (np.power(counts, args.factor))
    weights /= sum(weights)
    print(sum(weights))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("city-c gen")
    # parser.add_argument('factor', type=float, default=0.1, help="reweigt_factor")
    # parser.add_argument('worker', type=int, default=64, help="worker")
    # args, unparsed = parser.parse_known_args()
    # main(args)
    counts = [6108146841, 1008090855, 3779321151, 108633585, 145462041, 203315451, 34531191, 91567101, 2636198226, 191894334, 664377615, 201607155, 22334709, 1159508694, 44325015, 38987397, 38591796, 16337727, 68549292]
    weights = 1 / (np.power(counts, 0.4))
    weights /= sum(weights)
    # weights *
    print(weights)