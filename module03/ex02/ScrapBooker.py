import numpy as np


class ScrapBooker:
    def crop(self, array, dimensions, position=[0, 0]):
        """Crops the image as a rectangle via dim arguments (being the new height
        and width oof the image) from the coordinates given by position arguments.
        Args:
        array: numpy.ndarray
        dim: tuple of 2 integers.
        position: tuple of 2 integers.
        Returns:
        new_arr: the cropped numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if not (isinstance(array, np.ndarray)):
            return None
        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all([(isinstance(obj, int) and obj >= 0) for obj in dimensions])):
            return None
        height = len(array)
        width = len(array[0])
        if (height >= position[0] + dimensions[0] and width >= position[1] + dimensions[1]):
            res = array[position[0]:position[0] + dimensions[0],
                        position[1]:position[1] + dimensions[1]]
            return res
        return array

    def thin(self, array, n, axis):
        """
        Deletes every n-th line pixels along the specified axis (0: vertical, 1: horizontal)
        Args:
        array: numpy.ndarray.
        n: non null positive integer lower than the number of row/column of the array
        (depending of axis value).
        axis: positive non null integer.
        Returns:
        new_arr: thined numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if not (isinstance(array, np.ndarray)):
            return None
        axis = 0 if axis == 1 else 1
        d = range(n - 1, array.shape[axis], n)
        return np.delete(array, d, axis)

    def juxtapose(self, array, n, axis):
        """
        Juxtaposes n copies of the image along the specified axis.
        Args:
        array: numpy.ndarray.
        n: positive non null integer.
        axis: integer of value 0 or 1.
        Returns:
        new_arr: juxtaposed numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if not (isinstance(n, int)):
            return None
        if not (isinstance(axis, int) and (axis == 0 or axis == 1)):
            return None
        if not (isinstance(array, np.ndarray)):
            return None
        if n < 0:
            return None
        res = array
        for i in range(1, n):
            res = np.concatenate((array, res), axis)
        return res

    def mosaic(self, array, dimensions):
        """
        Makes a grid with multiple copies of the array. The dim argument specifies
        the number of repetition along each dimensions.
        Args:
        array: numpy.ndarray.
        dim: tuple of 2 integers.
        Returns:
        new_arr: mosaic numpy.ndarray.
        None otherwise (combinaison of parameters not incompatible).
        Raises:
        This function should not raise any Exception.
        """
        if not (isinstance(array, np.ndarray)):
            return None
        if not (isinstance(dimensions, tuple) and len(dimensions) == 2
                and all([(isinstance(obj, int) and obj >= 0) for obj in dimensions])):
            return None
        res = []
        res = self.juxtapose(array, dimensions[0], 0)
        res = self.juxtapose(res, dimensions[1], 1)
        return res
