import numpy as np


class ScrapBooker:
    def crop(self, array, dimensions, position=[0, 0]):
        # crop the image as a rectangle with the given
        # dimensions (meaning, the new height and width for the image), whose top left corner is
        # given by the position argument. The position should be (0,0) by default. You have to
        # consider it an error (and handle said error) if dimensions is larger than the current image
        # size.
        height = len(array)
        width = len(array[0])
        if (height >= position[0] + dimensions[0] and width >= position[1] + dimensions[1]):
            res = array[position[0]:position[0] + dimensions[0],
                        position[1]:position[1] + dimensions[1]]
            return res
        print("Choose position + dimension <= size_img")
        return array

    def thin(self, array, n, axis):
        # delete every n-th pixel row along the specified axis (0 vertical,
        # 1 horizontal), example below.
        return np.delete(array, np.s_[::n], axis)

    def juxtapose(self, array, n, axis):
        # juxtapose n copies of the image along the specified axis
        # (0 vertical, 1 horizontal).
        res = array
        for i in range(1, n):
            res = np.concatenate((array, res), axis)
        return res

    def mosaic(self, array, dimensions):
        # make a grid with multiple copies of the array. The
        # dimensions argument specifies the dimensions (meaning the height and width) of the grid
        # (e.g. 2x3).
        res = []
        res = self.juxtapose(array, dimensions[0], 0)
        res = self.juxtapose(res, dimensions[1], 1)
        return res
