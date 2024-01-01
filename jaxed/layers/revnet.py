from functools import partial

import jax
import flax.linen as nn

import jaxed.internals
from jaxed.layers.utils import lower_submodule_to_function

class RevNetBlock(nn.Module):
  f: nn.Module
  g: nn.Module

  @nn.compact
  def __call__(self, x1, x2, f_aux_args=(), g_aux_args=()):
    ffunc, fparams = lower_submodule_to_function(self, 'f', (x1, *f_aux_args))
    gfunc, gparams = lower_submodule_to_function(self, 'g', (x2, *g_aux_args))

    @jax.custom_ivjp
    def fwd_res_block(x1, x2, fparams, gparams):
      fout = ffunc((x2, *f_aux_args), fparams)
      fout, auxfout = split_if_multiple_outputs(fout)
      y1 = fout + x1
      gout = gfunc((y1, *g_aux_args), gparams)
      gout, auxgout = split_if_multiple_outputs(gout)
      y2 = gout + x2
      return y1, y2, auxfout, auxgout

    @fwd_res_block.defivjp
    def rev_block_ivjp(xs, ys, dys):
      x1, x2, fparams, gparams = xs
      del x1, x2 # We're going to reconstruct this and have these values pruned by JIT optimization
      (y1, y2, auxfout, auxgout) = ys
      (dy1, dy2, dauxfout, dauxgout) = dys

      dgo, dx2 = dy2, dy2
      go, gvjp = jax.vjp(gfunc, (y1, *g_aux_args), gparams)
      go, goaux = split_if_multiple_outputs(go)
      dgofull = join_if_multiple_outputs(dgo, dauxgout)
      ddy1, dgparams = gvjp(dgofull)
      dy1 += ddy1[0]
      del gvjp
      x2 = y2 - go

      dfo, dx1 = dy1, dy1
      fo, fvjp = jax.vjp(ffunc, (x2, *f_aux_args), fparams)
      fo, foaux = split_if_multiple_outputs(fo)
      dfofull = join_if_multiple_outputs(dfo, dauxfout)

      ddx2, dfparams = fvjp(dfofull)
      dx2 += ddx2[0]
      del fvjp
      x1 = y1 - fo

      return (x1, x2, fparams, gparams), (dx1, dx2, dfparams, dgparams)

    y1, y2, auxfout, auxgout = jax.invertible(fwd_res_block)(x1, x2, fparams, gparams)

    if len(auxfout)==0 and len(auxgout)==0:
        return y1, y2
    else:
        return y1, y2, auxfout, auxgout

class _RevNetBlockRef(nn.Module):
  f: nn.Module
  g: nn.Module
  def __call__(self, x1, x2, f_aux_args=(), g_aux_args=()):
    y1 = self.f(x2, *f_aux_args) + x1
    y2 = self.g(y1, *g_aux_args) + x2
    return y1, y2

def split_if_multiple_outputs(x):
  if isinstance(x, tuple):
    return x[0], x[1:]
  else:
    return x, ()

def join_if_multiple_outputs(a, b):
  if len(b) > 0:
    return (a, *b)
  else:
    return a
