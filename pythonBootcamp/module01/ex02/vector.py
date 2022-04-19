from copy import deepcopy
# https://docs.python.org/3/library/copy.html


class Vector:
    @staticmethod
    def shape_check(arr: list) -> bool:
        return all([isinstance(obj, float) for obj in arr])

    @staticmethod
    def shape_check_uno(arr: list) -> bool:
        return all([isinstance(obj, list) and
                    len(obj) == 1 and isinstance(obj[0], float)
                    for obj in arr])

    def __init__(self, a):
        if isinstance(a, (list, int, tuple)) == False:
            raise ValueError(
                "Error")
        if isinstance(a, int) and a < 0:
            raise ValueError("Error")
        if isinstance(a, tuple) and ((len(a) != 2) or
                                     not all([isinstance(obj, int) for obj in a]) or
                                     (a[0] >= a[1])):
            raise ValueError("Error")

        values = []
        shape = ()
        if (isinstance(a, list)):
            if self.shape_check(a):
                values = deepcopy(a)
                shape = (1, len(a))
            elif self.shape_check_uno(a):
                values = deepcopy(a)
                shape = (len(a), 1)
            else:
                raise ValueError("Error")
        elif isinstance(a, int):
            values = [[float(nb)] for nb in range(a)]
            shape = (a, 1)
        elif isinstance(a, tuple):
            values = [[float(nb)] for nb in range(*a)]
            shape = (a[1] - a[0], 1)
        else:
            raise Exception("Error")
        self.values = values
        self.shape = shape

    def __str__(self) -> str:
        return (f"Vector({self.values})")

    def get_value(self, index: int) -> float or int:
        if self.shape[0] < index and self.shape[1] < index:
            raise ValueError("Error")
        if isinstance(self.values[index], float):
            return self.values[index]
        return self.values[index][0]

    def __add__(self, other):
        if not (isinstance(other, Vector) and (other.shape == self.shape)):
            raise ValueError("Error")
        res = []
        if self.shape == (1, 1):
            res.append(self.get_value(0) + other.get_value(0))
        elif self.shape[0] > 1:
            for a, b in zip(self.values, other.values):
                res.append([a[0] + b[0]])
        else:
            for a, b in zip(self.values, other.values):
                res.append(a + b)
        return Vector(res)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not (isinstance(other, Vector) and (other.shape == self.shape)):
            raise ValueError("Error")
        res = []
        if self.shape == (1, 1):
            res.append(self.get_value(0) - other.get_value(0))
        elif self.shape[0] > 1:
            for a, b in zip(self.values, other.values):
                res.append([a[0] - b[0]])
        else:
            for a, b in zip(self.values, other.values):
                res.append(a - b)
        return Vector(res)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError("Error")
        if float(other) == 0.0:
            raise ValueError("Error")
        res = []
        if self.shape == (1, 1):
            res.append(self.get_value(0) / other)
        elif self.shape[0] > 1:
            for a in self.values:
                res.append([a[0] / other])
        else:
            for a in self.values:
                res.append(a / other)
        return Vector(res)

    def __rtruediv__(self, other):
        raise ValueError('Error')

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError("Error")
        res = []
        if self.shape == (1, 1):
            res.append(self.get_value(0) * other)
        elif self.shape[0] > 1:
            for a in self.values:
                res.append([a[0] * other])
        else:
            for a in self.values:
                res.append(a * other)
        return Vector(res)

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other) -> float or int:
        if not (isinstance(other, Vector) and
                (self.shape == other.shape or self.shape == other.shape[::-1])):
            raise ValueError("Error")
        res = 0
        length = len(self.values)
        for i in range(length):
            res += self.get_value(i) * other.get_value(i)
        return res

    def T(self):
        res = []
        for x in self.values:
            if isinstance(x, float):
                res.append([x])
            else:
                res.append(x[0])
        return Vector(res)
