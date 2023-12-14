from functools import partial

import jax
import flax.linen as nn

class RevNetBlock(nn.Module):
  f: nn.Module
  g: nn.Module

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

    self.rev_block = rev_block

  def __call__(self, x1, x2):
    return self.rev_block(x1, x2)
