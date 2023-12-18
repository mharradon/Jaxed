from functools import partial

import jax
import flax.linen as nn

class RevNetBlock(nn.Module):
  f: nn.Module
  g: nn.Module
  use_inverse: bool = True

"""
  def setup(self):
    @jax.custom_ivjp
    def rev_block(x1, x2):
      y1 = self.f(x2) + x1
      y2 = self.g(y1) + x2
      return y1, y2

    @rev_block.defivjp
    def rev_block_ivjp(xs, ys, dys, **kwargs):
      (y1, y2) = ys
      (dy1, dy2) = dys

      dgo, dx2 = dy2, dy2
      go, gvjp = jax.vjp(self.g, y1)
      dy1 += gvjp(dgo)[0]
      del gvjp
      x2 = y2 - go

      dfo, dx1 = dy1, dy1
      fo, fvjp = jax.vjp(self.f, x2)
      dx2 += fvjp(dfo)[0]
      del fvjp
      x1 = y1 - fo

      return (x1, x2), (dx1, dx2)
"""


  def __call__(self, x1, x2):
    def ffunc(fx, params):
      return self.f.apply({'params': params}, *fx)

    def gfunc(gx, params):
      return self.g.apply({'params': params}, *gx)

    @jax.custom_ivjp
    def fwd_res_block(x1, x2, fparams, gparams):
      y1 = ffunc(x2, fparams) + x1
      y2 = gfunc(y1, gparams) + x2
      return y1, y2

    @fwd_res_block.defivjp
    def rev_block_ivjp(xs, ys, dys):
      x1, x2, fparams, gparams = xs
      del x1, x2
      (y1, y2) = ys
      (dy1, dy2) = dys

      dgo, dx2 = dy2, dy2
      go, gvjp = jax.vjp(gfunc, y1, gparams)
      ddy1, dgparams = gvjp(dgo)
      dy1 += ddy1
      del gvjp
      x2 = y2 - go

      dfo, dx1 = dy1, dy1
      fo, fvjp = jax.vjp(ffunc, x2, fparams)

      ddx2, dfparams = fvjp(dfo)
      dx2 += ddx2
      del fvjp
      x1 = y1 - fo

      return (x1, x2, fparams, gparams), (dx1, dx2, dfparams, dgparams)

    if self.use_inverse:
      fwd_res_block = jax.invertible(fwd_res_block)

    fparams = self.param('f', self.f.init, jax.random.PRNGKey(0), x1)
    gparams = self.param('g', self.g.init, jax.random.PRNGKey(0), x2)

    return fwd_res_block(x1, x2, fparams, gparams)
