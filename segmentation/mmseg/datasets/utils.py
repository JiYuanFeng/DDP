import numpy as np
from mmseg.core import get_palette


def plot_mask_with_palette(mask_array, palette_name="cityscapes"):
    color_seg = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    palette = get_palette(palette_name)
    for label, color in enumerate(palette):
        color_seg[mask_array == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    import matplotlib.pyplot as plt
    plt.imshow(color_seg)
    plt.show()
