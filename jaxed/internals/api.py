from typing import Callable

import jax

from . import invertible_ad as iad

def invertible(fun: Callable) -> Callable:
  """Asserts that the decorated function is invertible.
  Applying reverse-mode AD to a decorated function will use a more memory efficient
  procedure than usual, which will reconstruct the necessary intermediate values
  by inverting the function. Note that this might degrade the numerical accuracy of
  obtained gradients if the inverse is unstable.
  Args:
    fun: The function assumed to be invertible.
  """
  return iad.invertible(fun)

jax.invertible = invertible
