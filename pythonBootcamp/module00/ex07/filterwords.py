import re
from sys import argv as av


def filterwords(text, min_size):
    return [word for word in re.findall(r"[\w']+", text) if len(word) > min_size]


if len(av) != 3 or av[2].isdigit() == False or int(av[2]) <= 0:
    print("ERROR")
else:
    print(filterwords(av[1], int(av[2])))
