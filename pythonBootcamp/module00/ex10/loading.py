import sys
from time import sleep
import time

start_time = time.time()
listy = range(3333)
ret = 0


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def ft_progress(listy):
    total = len(listy)
    unit = int(total / 20)
    for i in listy:
        elapsed_time = ((time.time()) - (start_time))
        eta = ((elapsed_time) * (total - i) / (i + 1))
        print("ETA: ", ' ' if (eta // 10) < 1 else '', str(truncate(eta, 2)) + "s ",
              "[ " if i < (total // 10) else "[", str(((i * 100) // total)) + '%' + ']' +
              '[' + (i // unit) * '=' + '>' + ((total - i - 1) // unit) * ' ' + '] ' +
              str(i) + '/' + str(total) +
              " | elapsed time " + str(truncate(elapsed_time, 2)) + "s  ",
              end="\r")
        yield i
    print("")


for elem in ft_progress(listy):
    # ret += (elem + 3) % 5
    ret += elem
    sleep(0.005)
print("...")
print(ret)
