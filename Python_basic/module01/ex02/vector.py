class Vector:
    def __init__(self, values):
        assert isinstance(values, list) or isinstance(
            values, int) or isinstance(values, tuple), "Invalid value"
        self.values = list()
        if type(values) == list:
            for c in values:
                assert isinstance(c, float), "Error, only float"
            self.values = values
        elif type(values) == int:
            for c in range(values):
                self.values.append(float(c))
        elif type(values) == tuple:
            assert len(values) == 2, "Invalid nb of values"
            for c in range(values[0], values[1]):
                self.values.append(float(c))
        self.size = len(self.values)

    def __str__(self):
        txt = "(Vector {values})".format(values=self.values)
        return txt

    def __repr__(self) -> str:
        pass

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        assert isinstance(other, int) or isinstance(
            other, float) or isinstance(other, Vector), "other value is invalid"
        if type(other) == int or type(other) == float:
            tab = list()
            for c in self.values:
                tab.append(c + other)
        elif type(other) == Vector:
            assert self.size == other.size, "Invalid size"
            tab = list()
            i = 0
            while i < self.size:
                tab.append(self.values[i] + other.values[i])
                i += 1
            return (Vector(tab))

    def __rsub__(self, other):
        assert isinstance(other, Vector), "Other is not a Vector."
        tab = list()
        i = 0
        while i < self.size:
            tab.append(other.values[i] - self.values[i])
            i += 1
        return(Vector(tab))

    def __sub__(self, other):
        assert isinstance(other, int) or isinstance(
            other, float) or isinstance(other, Vector), "other value is invalid"
        if type(other) == int or type(other) == float:
            tab = list()
            for c in self.values:
                tab.append(c - other)
        elif type(other) == Vector:
            assert self.size == other.size, "Invalid size"
            tab = list()
            i = 0
            while i < self.size:
                tab.append(self.values[i] - other.values[i])
                i += 1
            return (Vector(tab))

    def __rtruediv__(self, other):
        assert isinstance(other, int) or isinstance(other, float) or isinstance(
            other, Vector), "Other is not a int or a float or a Vector."
        tab = list()
        i = 0
        while i < self.size:
            if other.values[i] != 0:
                tab.append(self.values[i] / other.values[i])
            else:
                print('Err: div by 0!')
                return None
            i += 1
        return(Vector(tab))

    def __truediv__(self, other):
        assert isinstance(other, int) or isinstance(other, float) or isinstance(
            other, Vector), "Other is not a int or a float or a Vector."
        if type(other) == int or type(other) == float:
            tab = list()
            for value in self.values:
                tab.append(value / other)
        elif type(other) == Vector:
            tab = list()
            i = 0
            while i < self.size:
                if other.values[i] != 0:
                    tab.append(self.values[i] / other.values[i])
                else:
                    print('Err: div by 0!')
                    return None
                i += 1
        return(Vector(tab))

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(
            other, float) or isinstance(other, Vector), "other value is invalid"
        if type(other) == int or type(other) == float:
            tab = list()
            for c in self.values:
                tab.append(c * other)
        elif type(other) == Vector:
            assert self.size == other.size, "Invalid size"
            tab = list()
            i = 0
            while i < self.size:
                tab.append(self.values[i] * other.values[i])
                i += 1
            return (Vector(tab))
