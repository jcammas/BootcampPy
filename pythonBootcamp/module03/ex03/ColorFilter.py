from ImageProcessor import ImageProcessor
import numpy as np


class ColorFilter:
    def __init__(self) -> None:
        pass

    def invert(array):
        return 255 - array[:, :, :3]

    def to_blue(array):
        array[:, :, 0] = 0
        array[:, :, 1] = 0
        return array

    def to_green(array):
        array[:, :, 0] *= 0
        array[:, :, 2] *= 0
        return array

    def to_red(array):
        array[:, :, 1] = 0
        array[:, :, 2] = 0
        return array

    def to_celluloid(array):
        array[array < 64] = 0
        array[(array > 64) & (array < 128)] = 64
        array[array > 128] = 128

        return array

    def to_grayscale(array, _filter):
        if _filter == "m" or _filter == "mean":
            array[:, :, 0:3] = np.sum(
                array[:, :, 0:3] / 3, axis=2, keepdims=True).astype(array.dtype)
            return array
        elif _filter == "weighted" or _filter == "w":
            array[:, :, 0:3] = np.sum([array[:, :, 0:1] * 0.299, 0.587 * array[:, :, 1:2], 0.114 * array[:, :, 2:3]],
                                      axis=0)
            return array
