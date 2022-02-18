from typing import Type


class Vector:
    def __init__(self, arg):
        self.values = None
        if type(arg) is int and arg > 0:
            self.values = [float(value) for value in range(arg)]
        if type(arg) is list and [value for value in arg if type(value) is not float] == []:
            self.values = [float(value) for value in arg]
        if type(arg) is tuple and len(arg) == 2 and [value for value in arg if type(value) is not int] == []:
            self.values = [float(value) for value in range(arg[0], arg[1])]
        if self.values == None:
            raise ValueError(
                "Value must be either a list of floats, a positive integer or a tupple of two positive numbers (start, end).")

    def __str__(self):
        return f"(Vector {self.values})"

    def __repr__(self):
        return f"values={self.values}"

    def __add__(self, a):
        sum = None
        if type(a) is int:
            sum = [value + a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value + a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform addition of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __radd__(self, a):
        sum = None
        if type(a) is int:
            sum = [value + a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value + a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform addition of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __sub__(self, a):
        sum = None
        if type(a) is int:
            sum = [value - a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value - a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform substraction of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __rsub__(self, a):
        sum = None
        if type(a) is int:
            sum = [value - a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value - a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform substraction of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __mul__(self, a):
        sum = None
        if type(a) is int:
            sum = [value * a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value * a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform multiplication of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __rmul__(self, a):
        sum = None
        if type(a) is int:
            sum = [value * a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values):
            sum = [value * a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform multiplication of differently sized vectors or with anything else than a scalar")
        return Vector(sum)

    def __truediv__(self, a):
        sum = None
        if type(a) is int and a != 0:
            sum = [value * a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values) and [value for value in a.values if value == 0] == []:
            sum = [value * a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform true division of differently sized vectors or with anything else than a scalar. Can not divide by zero")
        return Vector(sum)

    def __rtruediv__(self, a):
        sum = None
        if type(a) is int and a != 0:
            sum = [value * a for value in self.values]
        if type(a) is Vector and len(a.values) == len(self.values) and [value for value in a.values if value == 0] == []:
            sum = [value * a.values[i] for i, value in enumerate(self.values)]
        if sum == None:
            raise TypeError(
                "Can not perform true division of differently sized vectors or with anything else than a scalar. Can not divide by zero")
        return Vector(sum)
