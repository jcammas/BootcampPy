import sys


def checkNumber(arg):
    nb = int(arg)
    s = ''
    if nb == 0:
        s = "I'm Zero."
    elif nb % 2 == 0:
        s = "I'm Even."
    elif nb % 2 == 1:
        s = "I'm Odd."
    else:
        s = "ERROR"
    return s


if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.isdigit():
            s = checkNumber(arg)
            print(s)
        else:
            print("ERROR")
    elif len(sys.argv) > 2:
        print("ERROR")
