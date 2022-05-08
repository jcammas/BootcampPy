import numpy as np
import math


class TinyStatistician():
    @staticmethod
    def mean(x):
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        _sum = 0.0
        for i in x:
            _sum += i
        return float(_sum) / len(x)

    @staticmethod
    def median(x):
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        if len(x) == 0:
            return None
        try:
            l = sorted(x)
        except TypeError:
            return None
        size = len(x)
        if size % 2 == 1:
            return sorted(x)[size // 2]
        else:
            x = sorted(x)
            a = x[(size // 2) - 1]
            b = x[size // 2]
            res = (a + b) / 2
            return res

    @staticmethod
    def percentile(data, perc: int):
        if not isinstance(data, list) and not isinstance(data, np.ndarray):
            return None
        if perc < 1 or perc > 100:
            return None
        try:
            data.sort()
        except TypeError:
            return None
        size = len(data)
        idx = int(math.ceil((size * perc) / 100)) - 1
        return sorted(data)[idx]

    @staticmethod
    def quartile(x):
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None

        q1 = TinyStatistician.percentile(x, 25)
        q3 = TinyStatistician.percentile(x, 75)

        return [q1, q3]

    @staticmethod
    def var(x):
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        mean = TinyStatistician.mean(x)

        res = 0.0
        for i in x:
            res += float(i - mean) ** 2.0

        res = res / float(len(x))
        return res

    @staticmethod
    def std(x):
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            return None
        try:
            x.sort()
        except TypeError:
            return None
        if len(x) == 0:
            return None
        res = TinyStatistician.var(x) ** 0.5
        return res


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
