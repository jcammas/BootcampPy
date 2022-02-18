import sys


def rev_alpha(*args):
    tmp = ""
    for arg in args:
        tmp += " ".join(arg)
    res = tmp.swapcase()
    return res[::-1]


if __name__ == "__main__":
    print(rev_alpha(sys.argv[1:]))

# https://realpython.com/reverse-string-python/
