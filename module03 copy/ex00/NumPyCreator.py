import numpy as np

class NumPyCreator:
    def __init__(self, values = []):
		pass

    def from_list(self, lst):
        if not isinstance(lst, list):
            return None
        if len(lst) > 1 and type(lst[0]) is list:
            for i in lst:
                if len(i) != len(lst[0]):
                    return None
        return np.asarray(lst)

    def from_tuple(self, tpl):
        if not isinstance(tpl, tuple):
            return None
        if len(tpl) > 1 and type(tpl[0]) is tuple:
            for i in tpl:
                if len(i) != len(tpl[0]):
                    return None
        return np.asarray(tpl)

	# def from_iterable(self, itr):
	# 	if not hasattr(itr, __iter__):
    #         return None
	# 	return np.fromiter(itr, float)

	def from_shape(self, shape, value = 0):
		#returns an array filled with the same value.
		#The first argument is a tuple which specifies the shape of the array,
			#and the second argument specifies the value of all the elements.
		#This value must be 0 by default.
		return np.full(shape, value)

	def random(self, shape):
		#returns an array filled with random values.
		#It takes as an argument a tuple which specifies the shape of the array.
		return np.random.uniform(size=shape)

	
    def identity(self, n):
        if not isinstance(n, int) or n < 0:
            return None
        return np.identity(n)
