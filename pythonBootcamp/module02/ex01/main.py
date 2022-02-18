def what_are_the_vars(*args, **kwargs) -> object:
    """..."""
    obj = ObjectC()
    if not len(args) and not len(kwargs):
        return obj

    svr = "var_"
    count = 0
    svr_set = set()
    for i in args:
        obj.__setattr__(svr + str(count), i)
        svr_set.add(svr + str(count))
        count += 1

    for k, v in kwargs.items():
        if k in svr_set:
            return None
        obj.__setattr__(k, v)

    return obj


class ObjectC(object):
    def __init__(self):
        pass


def doom_printer(obj):
    if obj is None:
        print("ERROR")
        print("end")
        return
    for i in dir(obj):
        if i[0] != "_":
            value = getattr(obj, i)
            print("{}: {}".format(i, value))
    print("end")


if __name__ == "__main__":
    obj = what_are_the_vars(7)
    doom_printer(obj)
    obj = what_are_the_vars(None, [])
    doom_printer(obj)
    obj = what_are_the_vars("ft_lol", "Hi")
    doom_printer(obj)
    obj = what_are_the_vars()
    doom_printer(obj)
    obj = what_are_the_vars(12, "Yes", [0, 0, 0], a=10, hello="world")
    doom_printer(obj)
    obj = what_are_the_vars(42, a=10, var_0="world")
    doom_printer(obj)
    obj = what_are_the_vars(42, "Yes", a=10, var_2="world")
    doom_printer(obj)
