from ImageProcessor import ImageProcessor
import numpy as np


class ColorFilter:
    def __init__(self) -> None:
        pass

    def invert(self, array):
        res = 1 - array
        res[..., 3:] = array[..., 3:]
        return res

    def to_blue(self, array):
        array[:, :, 0] = 0
        array[:, :, 1] = 0
        return array

    def to_green(self, array):
        array[:, :, 0] *= 0
        array[:, :, 2] *= 0
        return array

    def to_red(self, array):
        array[:, :, 1] = 0
        array[:, :, 2] = 0
        return array

    # def to_celluloid(self, array):
    #     array[array < 64] = 0
    #     array[(array > 64) & (array < 128)] = 64
    #     array[array > 128] = 128

    #     return array

    def to_grayscale(self, array, filter, **kwargs):
        if filter == "m" or filter == "mean":
            array[:, :, 0:3] = np.sum(
                array[:, :, 0:3] / 3, axis=2, keepdims=True).astype(array.dtype)
            return array
        elif filter == "weighted" or filter == "w":
            array[:, :, 0:3] = np.sum([array[:, :, 0:1] * 0.299, 0.587 * array[:, :, 1:2], 0.114 * array[:, :, 2:3]],
                                      axis=0)
            return array
