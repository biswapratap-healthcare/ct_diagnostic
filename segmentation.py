import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.segmentation import clear_border


def device_removal(im):
    upper_black_to_white_transition = False
    lower_black_to_white_transition = False
    for x in range(0, im.shape[0], 1):
        if upper_black_to_white_transition is False:
            all_false = not im[x].any()
            if all_false:
                continue
            else:
                upper_black_to_white_transition = True
        if upper_black_to_white_transition is True and lower_black_to_white_transition is False:
            all_false = not im[x].any()
            if all_false is False:
                continue
            else:
                lower_black_to_white_transition = True
        if upper_black_to_white_transition is True and lower_black_to_white_transition is True:
            for y in range(0, im.shape[1], 1):
                im[x, y] = False
    return im


def apply_mask(arr, mask):
    i_max = arr.shape[0]
    j_max = arr.shape[1]
    for i in range(i_max):
        for j in range(j_max):
            v = mask[i, j]
            if v == np.bool_(False):
                arr[i, j] = 0
            else:
                arr[i, j] = 255
    return arr


def get_segmented_lungs(im, plot=False):
    if plot is True:
        f, plots = plt.subplots(2, 1, figsize=(5, 40))
    if plot is True:
        plots[0].axis('off')
        plots[0].imshow(im, cmap=plt.cm.bone)
    binary = im < 604
    cleared = clear_border(binary)
    eroded = morphology.erosion(cleared, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))
    final_mask = device_removal(dilation)
    final_image = apply_mask(im, final_mask)
    if plot is True:
        plots[1].axis('off')
        plots[1].imshow(final_image, cmap=plt.cm.bone)
    return final_image
