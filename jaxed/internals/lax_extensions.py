import jax
from jax._src.lax import lax
from jax import numpy as jnp
from jax import nn

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
iad.primitive_ivjps[lax.exp_p] = lambda x, y, ct, **kwargs: [[lax.log(y[0])], [ct[0] * y[0]]]
iad.definverse(lax.log_p, lambda r, x: lax.exp(r))
iad.definverse(lax.add_p, _add_inverse)
iad.definverse(lax.sub_p, _sub_inverse)
iad.definverse(lax.mul_p, _mul_inverse)
iad.definverse(lax.div_p, _div_inverse)

nn.selu = jax.custom_ivjp(nn.selu)
@nn.selu.defivjp
def selu_ivjp(x, y, dy):
  #x, y, dy = x[0], y[0], dy[0]
  scale = 1.0507009873554804934193349852946
  alpha = 1.6732632423543772848170429916717

  scale_alpha = scale * alpha

  ypos = y > 0

  x = jnp.where(ypos,
                y / scale,
                jnp.log((y / scale_alpha) + 1))

  # d selu(x) / dx
  # where x <= 0
  # = scale_alpha * exp(x)
  # = selu(x) + scale_alpha
  dx = jnp.where(ypos,
                 dy * scale,
                 y + scale_alpha)

  return x, dx
