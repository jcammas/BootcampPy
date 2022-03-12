import math
import numpy as np


class TinyStatistician():
    def mean(x):
        """computes the mean of x, using a for-loop and returns the mean as a float,
        otherwise None. This method should not raise any exception.
        Given a vector x of dimension m, the mathematical formula of its mean is:
        µ = Pm  xi
            i=1
        -----------
            m
        """
        if len(x) == 0:
            return None
        sum = 0.0
        nb = 0
        for elem in x:
            try:
                sum += elem
                nb += 1
            except:
                return None
        return sum/nb

    def median(x):
        """computes the median (also called the 50th percentile) of x and returns
           it as a float, otherwise None. This method should not raise any exception."""
        if len(x) == 0:
            return None
        l = sorted(x)
        lent = len(l)
        if (lent % 2) == 0:
            m = int(lent / 2)
            res = l[m]
        else:
            m = int(float(lent / 2) - 0.5)
            res = l[m]
        return res

    def percentile(data, perc: int):
        """get the pth percentile of x, and returns the percentile as a float,
        otherwise None. This method should not raise any exception."""
        size = len(data)
        return sorted(data)[int(math.ceil((size * perc) / 100)) - 1]

    def quartile(x):
        """computes the 1st and 3rd quartiles (also called the 25th percentile and
        the 75th percentile) of x, and returns the quartiles as a list of 2 floats, otherwise
        None. This method should not raise any exception."""
        if len(x) == 0:
            return None
        q1 = TinyStatistician.percentile(x, 25)
        q3 = TinyStatistician.percentile(x, 75)

        return [q1, q3]

    def var(x):
        """computes the variance of x and returns it as a float, otherwise None. This
        method should not raise any exception."""
        if len(x) == 0:
            return None
        m = TinyStatistician.mean(x)
        nb = 0
        v = np.array([])
        for elem in x:
            gap = (elem - m) ** 2
            v = np.append(v, gap)
        return TinyStatistician.mean(v)

    def std(x):
        """computes the standard deviation of x, and returns it as a float, otherwise
        None. This method should not raise any exception."""
        return math.sqrt(TinyStatistician.var(x))


X = np.array([1, 42, 300, 10, 59])
print(TinyStatistician.mean(X))
print(TinyStatistician.median(X))
print(TinyStatistician.quartile(X))
print(TinyStatistician.percentile(X, 10))
print(TinyStatistician.percentile(X, 28))
print(TinyStatistician.percentile(X, 83))
print(TinyStatistician.var(X))
print(TinyStatistician.std(X))
