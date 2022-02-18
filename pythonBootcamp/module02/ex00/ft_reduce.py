def ft_reduce(function_to_apply, iterable):
    """Reducing consists of applying a reduction function to an iterable 
       to produce a single cumulative value."""
    if not len(iterable):
        raise ValueError("Error")
    tmp = iter(iterable)
    nxt = next(tmp)
    for i in tmp:
        nxt = function_to_apply(nxt, i)
    return nxt


lst = ["H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
print(ft_reduce(lambda u, v: u + v, lst))
