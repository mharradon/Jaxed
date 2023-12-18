from time import time

def timefunc(f, *args, N=40):
    tic = time()
    for i in range(N):
      _ = f(*args)
    avg_runtime = (time() - tic) / N
    return avg_runtime
