# Copyright (c) 2015-2021 Anish Athalye. Released under GPLv3.

import os
import math
import re
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import numpy as np

from stylize import stylize

def fmt_imsave(fmt, iteration):
    if re.match(r"^.*\{.*\}.*$", fmt):
        return fmt.format(iteration)
    elif "%" in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))


def main():
    # https://stackoverflow.com/a/42121886
    key = "TF_CPP_MIN_LOG_LEVEL"
    if key not in os.environ:
        os.environ[key] = "2"

    content_image = imread("images_cnn/origin1.jpg")
    style_images = [imread("images_cnn/style1.jpg")]

    style_blend_weights = [1.0 / len(style_images) for _ in style_images]

    try:
        imsave("images_cnn/result8.jpg", np.zeros((500, 500, 3)))
    except:
        raise IOError(
            "result is not writable or does not have a valid file "
            "extension for an image file"
        )

    loss_arrs = None
    for iteration, image, loss_vals in stylize(
        network="imagenet-vgg-verydeep-19.mat",
        initial=None,
        initial_noiseblend=1.0,
        content=content_image,
        styles=style_images,
        preserve_colors=False,
        iterations=100,
        content_weight=5e0,
        content_weight_blend=1,
        style_weight=5e2,
        style_layer_weight_exp=1,
        style_blend_weights=style_blend_weights,
        tv_weight=1e2,
        learning_rate=1e1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        pooling="max",
        print_iterations=None,
        checkpoint_iterations=None,
    ):
        if (loss_vals is not None):
            if loss_arrs is None:
                itr = []
                loss_arrs = OrderedDict((key, []) for key in loss_vals.keys())
            for key, val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    imsave("images_cnn/result8.jpg", image)

def imread(path):
    img = np.array(Image.open(path)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def imresize(arr, size):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if isinstance(size, tuple):
        height, width = size
    else:
        width = int(img.width * size)
        height = int(img.height * size)
    return np.array(img.resize((width, height)))


if __name__ == "__main__":
    main()
