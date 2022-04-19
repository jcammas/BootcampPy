from loading import ft_progress
import time


if __name__ == "__main__":
    ret = 0

    listy = range(1000)
    for elem in ft_progress(listy):
        ret += (elem + 3) % 5
        time.sleep(0.01)
    print()
    print(ret)
