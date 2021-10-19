import sys


def mergeString(args):
    string = args[0]
    for arg in args[1:]:
        string = "{str} {arg}".format(str=string, arg=arg)
    return string


def reverseString(s):
    newString = ''.join(c.lower() if c.isupper() else c.upper() for c in s)
    return newString[::-1]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("")
    else:
        string = mergeString(sys.argv[1:])
        string = reverseString(string)
        print(string)
