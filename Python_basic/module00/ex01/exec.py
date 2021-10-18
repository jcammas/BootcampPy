import sys

string = ''
i = len(sys.argv)
while (i > 1):
    i -= 1
    letters = len(sys.argv[i])
    while (letters > 0):
        letters -= 1
        string += sys.argv[i][letters]
    if (i > 1):
        string += ' '
string = string.swapcase()
print(string)
