from time import time
import warnings
import sys
from os import devnull

from jaxlib.xla_extension import XlaRuntimeError

def timefunc(f, *args, N=40):
    _ = f(*args)
    tic = time()
    for i in range(N):
      _ = f(*args)
    avg_runtime = (time() - tic) / N
    return avg_runtime

class SuppressOOM:
    # TODO: Something better
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(devnull, 'w')

    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.original_stderr

