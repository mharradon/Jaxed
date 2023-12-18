from functools import partial

import numpy as np
from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import lax
import jaxlib
from jax._src import test_util as jtu

import flax.linen as nn

import jaxed
from jaxed.layers import RevNetBlock

class LayerTests(jtu.JaxTestCase):

  def test_revnet_block(self):
    rev_block, params = make_basic_rev_block(use_inverse=False)
    rev_block_inv, params_inv = make_basic_rev_block(use_inverse=True)

    def net(x1, x2):
      for i in range(2):
        x1, x2 = rev_block.apply({'params':params},
                                 x1, 
                                 x2)
      return x1, x2

    def net_inv(x1, x2):
      for i in range(2):
        x1, x2 = rev_block_inv.apply({'params':params_inv},
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

  @jtu.skip_on_devices("cpu")
  def test_perf(self):
    blocks = []
    blocks_inv = []
    N = 64
    #while True:
    for i in range(1):
      blocks.append(RevNetBlock(nn.Dense(N),
                                nn.Dense(N),
                                use_inverse=False))
      blocks_inv.append(RevNetBlock(nn.Dense(N),
                                    nn.Dense(N),
                                    use_inverse=True))
      net = nn.Sequential(blocks + [jnp.sum])
      net_inv = nn.Sequential(blocks_inv + [jnp.sum])
      params_key = jax.random.key(0)
      params = net.init(params_key,
                        np.ones((N, N)),
                        np.ones((N, N)))
      params_inv = net_inv.init(params_key,
                                np.ones((N, N)),
                                np.ones((N, N)))

      def v_and_g(params):
        def fwd(p):
          rev_block.apply({'params': p},
                          np.ones((N, N)),
                          np.ones((N, N)))
        return jax.value_and_grad(fwd, argnums=(0,))(params)

      v_and_g(params)


def make_basic_rev_block(use_inverse):
  rev_block = RevNetBlock(SinLayer(), CosLayer(), use_inverse=use_inverse)

  params_key = jax.random.key(0)
  params = rev_block.init(params_key, 
                          np.ones((1,)), 
                          np.ones((1,)))

  rev_block.apply({'params':params},
                  np.ones((1,)), 
                  np.ones((1,)))

  return rev_block, params

class CosLayer(nn.Module):
  def __call__(self, x):
    return jnp.cos(x)

class SinLayer(nn.Module):
  def __call__(self, x):
    return jnp.sin(x)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
