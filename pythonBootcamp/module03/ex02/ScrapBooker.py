import numpy as np


class ScrapBooker:

    def __init__(self) -> None:
        pass

    @staticmethod
    def _errorArray(array: np.ndarray) -> bool:
        if not(isinstance(array, np.ndarray)):
            return False
        return True

    @staticmethod
    def _errorShape(shape: tuple) -> bool:
        if not (isinstance(shape, tuple) and len(shape) == 2 and all([isinstance(obj, int) and obj >= 0 for obj in shape])):
            return False
        return True

    def crop(self, array: type[np.array], dimensions: tuple, position=[0, 0]):
        # crop the image as a rectangle with the given
        # dimensions (meaning, the new height and width for the image), whose top left corner is
        # given by the position argument. The position should be (0,0) by default. You have to
        # consider it an error (and handle said error) if dimensions is larger than the current image
        # size.
        if not (self._errorArray(array) and self._errorShape(dimensions) and self._errorShape(position)):
            return None

        if array.shape < dimensions:
            print("dimensions cannot be larger than the current image size.")
            return
        p1, p2 = position
        x, y = dimensions
        array = array[p1:x, p2:y]
        return array

    def thin(self, array: type[np.array], n: int, axis: int):
        # delete every n-th pixel row along the specified axis (0 vertical,
        # 1 horizontal), example below.
        if not (self._errorArray(array) and
                isinstance(n, int) and n > 0 and
                isinstance(axis, int) and (axis == 0 or axis == 1)):
            return None

        lst = []
        if not axis:
            for arr in array:
                for i in range(n - 1, len(arr), n - 1):
                    if i >= len(arr):
                        break
                    tmp = np.delete(arr, i)
                    arr = tmp
                lst.append(arr)
            return np.array(lst)
        else:
            for i in range(n - 1, array.shape[0], n - 1):
                if i >= array.shape[0]:
                    break
                array = np.delete(array, i, 0)
            return array

    def juxtapose(self, array, n, axis):
        # juxtapose n copies of the image along the specified axis
        # (0 vertical, 1 horizontal).
        if not (self._errorArray(array) and
                isinstance(n, int) and n > 0 and
                isinstance(axis, int) and (axis == 0 or axis == 1)):
            return None

        res = np.copy(array)
        for i in range(n - 1):
            res = np.concatenate((res, array), axis=axis)
        return res

    def mosaic(self, array, dimensions):
        # make a grid with multiple copies of the array. The
        # dimensions argument specifies the dimensions (meaning the height and width) of the grid
        # (e.g. 2x3).
        if not (self._errorArray(array) and self._errorShape(dimensions)):
            return None

        return np.array(np.tile(array, dimensions))
