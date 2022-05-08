class Matrix:
    def __init__(self, elements: list = None, shape: tuple = None) -> None:
        if isinstance(elements, list):
            self.data = elements.copy()
            self.shape = (len(elements), len(elements[0]))
        elif isinstance(shape, tuple):
            self.shape = shape
            self.data = [[0] * self.shape[1] for tmp in range(self.shape[0])]

    def T(self):
        matrix = []
        for i in range(len(self.data[0])):
            line = []
            for j in range(len(self.data)):
                line.append(self.data[j][i])
            matrix.append(line)
        return type(self)(matrix)

    def __add__(self, other):
        # add : vectors and matrices, can have errors with vectors and matrices.
        if isinstance(other, Matrix) and self.shape != other.shape:
            return
        res = Matrix(shape=(self.shape))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res.data[i][j] = self.data[i][j] + other.data[i][j]
        return res

    def __sub__(self, other):
        # sub : vectors and matrices, can have errors with vectors and matrices.
        if isinstance(other, Matrix) and self.shape != other.shape:
            return
        res = Matrix(shape=(self.shape))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res.data[i][j] = self.data[i][j] - other.data[i][j]
        return res

    def __truediv__(self, other):
        # div : only scalars.
        if not isinstance(other, (int, float)):
            return
        if other == 0:
            return None
        res = Matrix(shape=(self.shape))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res.data[i][j] = self.data[i][j] / other
        return res

    def __mul__(self, other):
        # mul : scalars, vectors and matrices , can have errors with vectors and matrices.
        # if we perform Matrix * Vector (dot product), return a Vector.
        res = None
        if isinstance(other, (int, float)):
            res = Matrix(shape=(self.shape))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    res.data[i][j] = self.data[i][j] * other
        elif isinstance(other, Matrix):
            common_len = self.shape[1]
            res = Matrix(shape=(self.shape[0], other.shape[1]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    res.data[i][j] = sum(
                        [self.data[i][k] * other.data[k][j] for k in range(common_len)])
        return res

    def __radd__(self, other):
        return Matrix.__add__(self, other)

    def __rsub__(self, other):
        return Matrix.__sub__(self, other)

    def __rmul__(self, other):
        return Matrix.__mul__(self, other)

    def __rtruediv__(self, other):
        return Matrix.__truediv__(self, other)

    def __str__(self) -> str:
        if self.shape[0] != 1 and self.shape[1] != 1:
            return "(Matrix %s)" % (self.data)
        else:
            return "(Vector %s)" % (self.data)

    def __repr__(self) -> str:
        pass


class Vector(Matrix):
    def __init__(self, data_or_shape):
        super(Vector, self).__init__(data_or_shape)
        if self.shape[0] != 1 and self.shape[1] != 1:
            return None

    def dot(self, other):
        if type(other) != Vector:
            return None
        if self.shape != other.shape:
            return None
        if self.shape[0] != 1:
            _self = self.T()
        else:
            _self = self
        data = []
        for i in _self.data[0]:
            nb = 0
            for ii in _self.data[0]:
                nb += i * ii
            data.append(nb)
        vector = Vector([data])
        if self.shape[0] != 1:
            vector = vector.T()
        return vector
