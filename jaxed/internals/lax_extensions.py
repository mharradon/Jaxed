from jax._src.lax import lax

from . import invertible_ad as iad

"""
Implementations for various inverse operators.
"""

def _add_inverse(r, x, y):
  xr = r - y
  yr = r - x
  return xr, yr

def _sub_inverse(r, x, y):
  # r = x - y
  xr = r + y
  yr = x - r
  return xr, yr

def _mul_inverse(r, x, y):
  xr = r / y
  yr = r / x
  return xr, yr

def _div_inverse(r, x, y):
  # r = x / y
  xr = r * y
  yr = x / r
  return xr, yr

iad.definverse(lax.exp_p, lambda r, x: lax.log(r))
iad.primitive_ivjps[lax.exp_p] = lambda x, y, ct: [[lax.log(y[0])], [ct[0] * y[0]]]
iad.definverse(lax.log_p, lambda r, x: lax.exp(r))
iad.definverse(lax.add_p, _add_inverse)
iad.definverse(lax.sub_p, _sub_inverse)
iad.definverse(lax.mul_p, _mul_inverse)
iad.definverse(lax.div_p, _div_inverse)

