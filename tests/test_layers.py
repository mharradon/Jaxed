from functools import partial
from itertools import count

import numpy as np
from absl.testing import absltest
from time import sleep

import jax
from jax import numpy as jnp
from jax import lax
import jaxlib
from jaxlib.xla_extension import XlaRuntimeError
from jax._src import test_util as jtu

import flax.linen as nn

import jaxed
from jaxed.layers import RevNetBlock, Invertible
from jaxed.layers.revnet import _RevNetBlockRef
from jaxed.utils import timefunc

class LayerTests(jtu.JaxTestCase):

  def test_revnet_block(self):
    rev_block, params = make_basic_rev_block(use_inverse=False)
    rev_block_inv, params_inv = make_basic_rev_block(use_inverse=True)

    def net(x1, x2):
      for i in range(2):
        x1, x2 = rev_block.apply(params,
                                 x1, 
                                 x2)
      return x1, x2

    def net_inv(x1, x2):
      for i in range(2):
        x1, x2 = rev_block_inv.apply(params_inv,
                                 x1, 
                                 x2)
      return x1, x2

    def reduce(f, x1, x2):
      y1, y2 = f(x1, x2)
      return np.sum(y1) + np.sum(y2)

    # FIXME: This breaks when argnums is left as default (i.e. 0), because JVP prunes
    #        zero tangents from call primitives.
    def v_and_g(x):
      return jax.value_and_grad(partial(reduce, net), argnums=(0, 1))(x, x + 2)

    def v_and_g_invertible(x):
      return jax.value_and_grad(partial(reduce, net_inv), argnums=(0, 1))(x, x + 2)

    x = np.ones((1,))
    self.assertAllClose(v_and_g_invertible(x),
                        v_and_g(x),
                        check_dtypes=True)
    self.assertAllClose(jax.jit(v_and_g_invertible)(x),
                        jax.jit(v_and_g)(x),
                        check_dtypes=True)

  def test_real_net(self):
    blocks = []
    N = 64
    normal_ad_block_limit = None
    rev_ad_block_limit = None
    blocks = []
    blocks_ref = []
    xs = (np.ones((N, N, N, N), dtype='float32'),
          np.ones((N, N, N, N), dtype='float32'))
    for _ in range(4):
      blocks.append(RevNetBlock(DenseWithAct(N),
                                DenseWithAct(N)))
      blocks_ref.append(_RevNetBlockRef(DenseWithAct(N),
                                        DenseWithAct(N)))

    net = nn.Sequential(blocks)
    net_ref = nn.Sequential(blocks_ref)
    params_key = jax.random.key(0)

    @jax.jit
    def v_and_g(params, x):
      def fwd(p, x):
        out = net.apply(p, *x)
        return jnp.sum(out[0] + out[1])
      return jax.value_and_grad(fwd, argnums=(0,))(params, x)

    params = net.init(params_key, *xs)
    params_ref = net_ref.init(params_key, *xs)

    v_and_g(params, xs)

  def test_drop_net(self):
    blocks = []
    N = 64
    normal_ad_block_limit = None
    rev_ad_block_limit = None
    blocks = []
    blocks_ref = []
    xs = (np.ones((N, N, N, N), dtype='float32'),
          np.ones((N, N, N, N), dtype='float32'))
    for _ in range(4):
      blocks.append(RevNetBlock(DenseWithDrop(N),
                                DenseWithDrop(N)))
      blocks_ref.append(_RevNetBlockRef(DenseWithDrop(N),
                                        DenseWithDrop(N)))

    net = SequentialAux(blocks, 2)
    net_ref = SequentialAux(blocks_ref, 2)
    params_key = jax.random.key(0)

    @jax.jit
    def v_and_g(params, x):
      def fwd(p, x):
        out = net.apply(p, *x, (True,), (True,), rngs={'dropout': jax.random.key(0)})
        return jnp.sum(out[0] + out[1])
      return jax.value_and_grad(fwd, argnums=(0,))(params, x)

    params = net.init(params_key, *xs, (False,), (False,))
    params_ref = net_ref.init(params_key, *xs, (False,), (False,))

    v_and_g(params, xs)

  @jtu.skip_on_devices("cpu")
  def test_perf(self):
    blocks = []
    N = 64
    normal_ad_block_limit = None
    rev_ad_block_limit = None
    blocks = [RevNetBlock(DenseWithAct(N),
                          DenseWithAct(N))]
    blocks_ref = [_RevNetBlockRef(DenseWithAct(N),
                                  DenseWithAct(N))]
    xs = (np.ones((N, N, N, N), dtype='float32'),
          np.ones((N, N, N, N), dtype='float32'))
    for i in range(8):
      for _ in range(2**i):
        blocks.append(RevNetBlock(DenseWithAct(N),
                                  DenseWithAct(N)))
        blocks_ref.append(_RevNetBlockRef(DenseWithAct(N),
                                          DenseWithAct(N)))
      num_blocks = len(blocks)

      net = nn.Sequential(blocks)
      net_ref = nn.Sequential(blocks_ref)
      params_key = jax.random.key(0)

      @jax.jit
      def v_and_g(params, x):
        def fwd(p, x):
          out = net.apply(p, *x)
          return jnp.sum(out[0] + out[1])
        return jax.value_and_grad(fwd, argnums=(0,))(params, x)

      @jax.jit
      def v_and_g_ref(params, x):
        def fwd(p, x):
          out = net_ref.apply(p, *x)
          return jnp.sum(out[0] + out[1])
        return jax.value_and_grad(fwd, argnums=(0,))(params, x)

      print(f"{num_blocks} Blocks:")

      params = net.init(params_key, *xs)
      params_ref = net_ref.init(params_key, *xs)
      if normal_ad_block_limit is None:
        try:
          print(f"Normal AD: {timefunc(v_and_g_ref, params_ref, xs, N=5)} s")
        except XlaRuntimeError as E:
          sleep(3) # Print after OOM spew
          print(f"Normal AD OOMd at {num_blocks} blocks! {E}")
          normal_ad_block_limit = num_blocks

      try:
        print(f"Reversible AD: {timefunc(v_and_g, params, xs, N=5)} s")
      except XlaRuntimeError as E:
        sleep(3) # Print after OOM spew
        print(f"Reversible AD OOMd at {num_blocks} blocks! {E}")
        rev_ad_block_limit = num_blocks
        break

    print(f"Reversible AD does not OOM until at least {2*num_blocks} blocks!")

def make_basic_rev_block(use_inverse):
  rev_block = RevNetBlock(SinLayer(), CosLayer())

  params_key = jax.random.key(0)
  params = rev_block.init(params_key, 
                          np.ones((1,)), 
                          np.ones((1,)))

  rev_block.apply(params,
                  np.ones((1,)), 
                  np.ones((1,)))

  return rev_block, params

class CosLayer(nn.Module):
  def __call__(self, x):
    return jnp.cos(x)

class SinLayer(nn.Module):
  def __call__(self, x):
    return jnp.sin(x)

class DenseWithAct(nn.Module):
  n: int
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.n)(x)
    x = jax.invertible(jax.nn.selu)(x)
    #x = jax.nn.selu(x)
    return x

class DenseWithDrop(nn.Module):
  n: int
  @nn.compact
  def __call__(self, x, training):
    x = nn.Dense(self.n)(x)
    x = nn.Dropout(0.2)(x, deterministic=not training)
    x = jax.invertible(jax.nn.selu)(x)
    #x = jax.nn.selu(x)
    return x

class SequentialAux(nn.Module):
  blocks: list
  num_aux_args: int
  @nn.compact
  def __call__(self, *args):
    args, aux = args[:-self.num_aux_args], args[-self.num_aux_args:]
    for block in self.blocks:
      if not isinstance(args, tuple):
        args = (args,)
      args = block(*args, *aux)

    return args


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
