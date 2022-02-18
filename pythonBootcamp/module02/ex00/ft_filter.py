def ft_filter(function_to_apply, iterable):
    """Filtering consists of applying a predicate or Boolean-valued function to an iterable 
       to generate a new iterable. Items in the new iterable are produced by filtering out any items 
       in the original iterable that make the predicate function return false."""

    return [i for i in iterable if function_to_apply(i)]


x = [1, 2, 3, 4, 5]
print(ft_filter(lambda dum: not (dum % 2), x))
