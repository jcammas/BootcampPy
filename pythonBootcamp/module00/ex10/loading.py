import time


def ft_progress(lst):
    loadingBar = 25
    LSTLEN = len(lst)
    launchControl = time.time()
    eta = 0
    delta = None
    refresh_rate = LSTLEN / 20
    for i, elem in enumerate(lst, 1):
        ratio = i / LSTLEN
        percentage = ratio * 100
        bar = ("=" * int(ratio * (loadingBar - 1))) + ">"
        current_time = time.time()
        elapsed = current_time - launchControl
        if delta is not None:
            eta = delta * (LSTLEN - i)
        print("\rETA: %.2fs [%3d%%][%-*s] %d/%d | elapsed time %.2fs"
              % (eta, percentage, loadingBar, bar, i, LSTLEN, elapsed), end="")
        yield (elem)
        if delta is None or (i % refresh_rate) == 0:
            delta = time.time() - current_time
