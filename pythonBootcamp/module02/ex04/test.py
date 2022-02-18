import src.my_minipack.progressbar
import src.my_minipack.logger
import time


listy = range(100)
ret = 0
print(type(src.my_minipack))
print(type(src.my_minipack.progressbar))
print(type(src.my_minipack.progressbar.ft_progress))
for elem in src.my_minipack.progressbar.ft_progress(listy):
    ret += (elem + 3) % 5
    time.sleep(0.01)
print()

print(type(src.my_minipack.logger.log))


# https://packaging.python.org/en/latest/tutorials/packaging-projects/
