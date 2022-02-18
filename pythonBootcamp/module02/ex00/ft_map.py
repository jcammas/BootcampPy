def ft_map(function_to_apply, iterable):
    """Mapping consists of applying a transformation function to an iterable 
       to produce a new iterable. Items in the new iterable are produced by calling 
       the transformation function on each item in the original iterable."""
    tab = []
    for i in iterable:
        tab.append(function_to_apply(i))
    return tab


x = [1, 2, 3, 4, 5]
print(ft_map(lambda dum: dum + 1, x))

# https://realpython.com/python-map-function/
