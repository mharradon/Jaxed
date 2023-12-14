from functools import partial

import numpy as np
from absl.testing import absltest

import jax
from jax import numpy as jnp
from jax import lax
import jaxlib
from jax._src import test_util as jtu

from jaxed.layers import RevNetBlock

class LayerTests(jtu.JaxTestCase):

  def test_revnet_block(self):
    rev_block = RevNetBlock(jnp.sin, jnp.cos)

    params_key = jax.random.key(0)
    params = rev_block.init(params_key, 
                            np.ones((1,)), 
                            np.ones((1,)))

    rev_block.apply({'params':params},
                    np.ones((1,)), 
                    np.ones((1,)))

    def net(x1, x2):
      for i in range(2):
        rev_block.apply({'params':params},
                        x1, 
                        x2)
      return x1, x2

    def reduce(f, x1, x2):
      y1, y2 = f(x1, x2)
      return np.sum(y1) + np.sum(y2)

    # FIXME: This breaks when argnums is left as default (i.e. 0), because JVP prunes
    #        zero tangents from call primitives.
    def v_and_g_invertible(x):
      return jax.value_and_grad(partial(reduce, jax.invertible(net)), argnums=(0, 1))(x, x + 2)

    def v_and_g(x):
      return jax.value_and_grad(partial(reduce, net), argnums=(0, 1))(x, x + 2)

    x = np.ones((1,))
    self.assertAllClose(v_and_g_invertible(x),
                        v_and_g(x),
                        check_dtypes=True)
    self.assertAllClose(jax.jit(v_and_g_invertible)(x),
                        jax.jit(v_and_g)(x),
                        check_dtypes=True)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
