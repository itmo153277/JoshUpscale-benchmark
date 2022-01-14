# -*- encoding: utf-8 -*-

"""Script for loading training data."""

import os
from typing import Tuple
import numpy as np
import cv2


def load_data(hr_folder: str, lr_folder: str) -> Tuple[np.array, np.array]:
    """Load training data from the folder."""
    hr_data = []
    lr_data = []

    for img_name in sorted(os.listdir(hr_folder)):
        hr_img = cv2.imread(os.path.join(
            hr_folder, img_name), cv2.IMREAD_COLOR)
        lr_img = cv2.imread(os.path.join(
            lr_folder, img_name), cv2.IMREAD_COLOR)
        assert hr_img is not None
        assert lr_img is not None
        hr_data.append(hr_img[:, :, ::-1])
        lr_data.append(lr_img[:, :, ::-1])
    hr_data = np.array(hr_data).astype(np.float32) / 255
    lr_data = np.array(lr_data).astype(np.float32) / 255
    return hr_data, lr_data


def convert_nhwc_to_nchw(data: np.array) -> np.array:
    """Convert data to NCHW."""
    return np.transpose(data, [0, 3, 1, 2])


def convert_nchw_to_nhwc(data: np.array) -> np.array:
    """Convert data to NHWC."""
    return np.transpose(data, [0, 2, 3, 1])
