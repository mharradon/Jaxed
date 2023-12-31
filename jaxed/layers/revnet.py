from functools import partial

import jax
import flax.linen as nn

import jaxed.internals
from jaxed.layers.utils import lower_submodule_to_function

class RevNetBlock(nn.Module):
  f: nn.Module
  g: nn.Module

  @nn.compact
  def __call__(self, x1, x2):
    ffunc, fparams = lower_submodule_to_function(self, 'f', (x1,))
    gfunc, gparams = lower_submodule_to_function(self, 'g', (x2,))

    @jax.custom_ivjp
    def fwd_res_block(x1, x2, fparams, gparams):
      y1 = ffunc((x2,), fparams) + x1
      y2 = gfunc((y1,), gparams) + x2
      return y1, y2

    @fwd_res_block.defivjp
    def rev_block_ivjp(xs, ys, dys):
      x1, x2, fparams, gparams = xs
      del x1, x2
      (y1, y2) = ys
      (dy1, dy2) = dys

      dgo, dx2 = dy2, dy2
      go, gvjp = jax.vjp(gfunc, (y1,), gparams)
      ddy1, dgparams = gvjp(dgo)
      dy1 += ddy1[0]
      del gvjp
      x2 = y2 - go

      dfo, dx1 = dy1, dy1
      fo, fvjp = jax.vjp(ffunc, (x2,), fparams)

      ddx2, dfparams = fvjp(dfo)
      dx2 += ddx2[0]
      del fvjp
      x1 = y1 - fo

      return (x1, x2, fparams, gparams), (dx1, dx2, dfparams, dgparams)

    return fwd_res_block(x1, x2, fparams, gparams)


class _RevNetBlockRef(nn.Module):
  f: nn.Module
  g: nn.Module
  def __call__(self, x1, x2):
    y1 = self.f(x2) + x1
    y2 = self.g(y1) + x2
    return y1, y2
