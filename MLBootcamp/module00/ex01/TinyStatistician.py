import math
import numpy as np


ALLOWED_TYPES = [int, float, np.int64, np.float64]


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
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        sum = 0.0
        nb = 0
        for elem in x:
            if type(elem) not in ALLOWED_TYPES:
                return None
            try:
                sum += elem
                nb += 1
            except:
                return None
        return sum/nb

    def median(x):
        """computes the median (also called the 50th percentile) of x and returns
           it as a float, otherwise None. This method should not raise any exception."""
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        if len(x) == 0:
            return None
        try:
            l = sorted(x)
        except TypeError:
            return None
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
        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            return None
        if perc < 1 or perc > 100:
            return None
        try:
            data.sort()
        except TypeError:
            return None
        return float(data[int(len(data) * (perc / 100))])

    def quartile(x):
        """computes the 1st and 3rd quartiles (also called the 25th percentile and
        the 75th percentile) of x, and returns the quartiles as a list of 2 floats, otherwise
        None. This method should not raise any exception."""
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        q1 = float(x[int(len(x) * 0.25)])
        q3 = float(x[int(len(x) * 0.75)])

        return [q1, q3]

    def var(x):
        """computes the variance of x and returns it as a float, otherwise None. This
        method should not raise any exception."""
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
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
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        return math.sqrt(TinyStatistician.var(x))


print('\nError management : ', end='\n\n')
a_error = [1]
b_error = []
c_error = [1, 42, 'b', 10, 59]
print(TinyStatistician.mean(a_error))
print(TinyStatistician.mean(b_error))
print(TinyStatistician.mean(c_error))
print(TinyStatistician.median(a_error))
print(TinyStatistician.median(b_error))
print(TinyStatistician.median(c_error))
print(TinyStatistician.percentile(a_error, 10))
print(TinyStatistician.percentile(b_error, -3))
print(TinyStatistician.percentile(c_error, 10))
print(TinyStatistician.quartile(a_error), np.quantile(a_error, (0.25, 0.75)))
print(TinyStatistician.quartile(b_error))
print(TinyStatistician.quartile(c_error))
print(TinyStatistician.std(a_error), np.std(a_error))
print(TinyStatistician.std(b_error))
print(TinyStatistician.std(c_error))
print(TinyStatistician.var(a_error), np.var(a_error))
print(TinyStatistician.var(b_error))
print(TinyStatistician.var(c_error))

print('\nValid management : ', end='\n\n')
X = np.array([1, 42, 300, 10, 59])
print(TinyStatistician.mean(X), np.mean(X))
print(TinyStatistician.median(X), np.median(X))
print(TinyStatistician.quartile(X), np.quantile(X, (0.25, 0.75)))
print(TinyStatistician.percentile(X, 10))
print(TinyStatistician.percentile(X, 28))
print(TinyStatistician.percentile(X, 83))
print(TinyStatistician.var(X), np.var(X))
print(TinyStatistician.std(X), np.std(X))
