class Matrix:
    def __init__(self, elem: list = None, shape: tuple = None) -> None:
        if isinstance(elem, list):
            self.data = elem.copy()
            self.shape = (len(elem), len(elem[0]))
        elif isinstance(shape, tuple):
            self.shape = shape
            self.data = [[0] * self.shape[1] for tmp in range(self.shape[0])]

    def T(self):
        """method which returns the transpose of the matrix
            Transpose of a matrix is obtained by changing rows to columns and columns to rows."""
        matrix = []
        for i in range(len(self.data[0])):
            line = []
            for j in range(len(self.data)):
                line.append(self.data[j][i])
            matrix.append(line)
        return type(self)(matrix)

    def __add__(self, other):
        """add : vectors and matrices, can have errors with vectors and matrices."""
        if isinstance(other, Matrix) and self.shape != other.shape:
            return
        return Matrix([[self.data[i][j] + other.data[i][j]
                        for j in range(0, len(self.data[i]))]
                       for i in range(0, len(self.data))])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """sub : only matrices of same dimensions."""
        if isinstance(other, Matrix) and self.shape != other.shape:
            return
        return Matrix([[self.data[i][j] - other.data[i][j]
                        for j in range(0, len(self.data[i]))]
                       for i in range(0, len(self.data))])

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        """ div : only scalars."""
        if not isinstance(other, (int, float)):
            return
        if other == 0:
            return
        m = []
        for i in range(len(self.data)):
            line = []
            for j in range(len(self.data[0])):
                elem = self.data[i][j] / other
                line.append(elem)
            m.append(line)
        return type(self)(m)

    def __rtruediv__(self, other):
        raise ValueError("Error")

    def __mul__(self, other):
        """ mul : scalars, vectors and matrices , can have errors with vectors and matrices,
            returns a Vector if we perform Matrix * Vector mutliplication."""
        if isinstance(other, (int, float)):
            m = []
            for i in range(len(self.data)):
                line = []
                for j in range(len(self.data[0])):
                    elem = self.data[i][j] * other
                    line.append(elem)
                m.append(line)
            return type(self)(m)
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                return
            return self.__mulmatrix__(other)
        else:
            return

    def __mulmatrix__(self, other):
        matrix = []
        for i in range(len(self.data)):
            line = []
            for j in range(len(other.data[0])):
                elem = 0
                for k in range(len(other.data)):
                    elem += self.data[i][k] * other.data[k][j]
                line.append(elem)
            matrix.append(line)
        m = Matrix(matrix)
        if 1 in m.shape:
            m = Vector(m.data)
        return m

    def __rmul__(self, other):
        if type(other) == int or type(other) == float:
            return self.__mul__(other)
        else:
            return

    def __str__(self) -> str:
        return "(Matrix %s)" % (self.data)


class Vector(Matrix):
    def __init__(self, data_or_shape):
        super(Vector, self).__init__(data_or_shape)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError(
                f"Vector shape can't be {self.shape}, use Matrix instead")

    def dot(self, other):
        if type(other) != Vector:
            raise TypeError(
                f"Dot product is only between Vectors, not with {type(other)}")
        if self.shape != other.shape:
            raise ValueError(
                f"Dot product cant be done on vector of different shape")

        if self.shape[0] != 1:
            a, b = self.T(), other.T()
        else:
            a, b = self, other

        data = []
        for i in a.data[0]:
            nb = 0
            for ii in b.data[0]:
                nb += i * ii
            data.append(nb)

        v = Vector([data])
        if self.shape[0] != 1:
            v = v.T()
        return v
