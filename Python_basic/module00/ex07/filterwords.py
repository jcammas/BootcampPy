import sys
import string

if (len(sys.argv) is not 3):
    sys.exit("ERROR")
if (sys.argv[1].isdigit()):
    sys.exit("ERROR")
if (not sys.argv[2].isdigit() or int(sys.argv[2]) <= 0):
    sys.exit("ERROR")
words = str(sys.argv[1])
nb = int(sys.argv[2])
tab = []
for c in string.punctuation:
    words = words.replace(c, '')
words = words.split()
for i in words:
    if (len(i) > nb):
        tab.append(i)

print(tab)
