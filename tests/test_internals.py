from functools import partial
from time import time, sleep

import numpy as np
import jax
from jax import numpy as jnp
from jax import lax
from absl.testing import absltest
import jaxlib

import jaxed

from jax._src import test_util as jtu


class InvertibleADTest(jtu.JaxTestCase):

  @jtu.ignore_warning(message="Values that an @invertible function closes")
  def test_invertible_basic(self):
    def f(x):
      return lax.mul(lax.mul(lax.exp(x), 4.), x)

    finv = jax.invertible(f)
    x = jnp.ones((5,))

    jaxpr = jax.make_jaxpr(lambda p, ct: jax.vjp(finv, p)[1](ct))(x, x)

    expected = """
    { lambda  ; a b.
      let c = exp a
          d = mul c 4.0
          e = mul d a
          f = mul b a
          g = div e a
          h = mul b g
          i = mul f 4.0
          j = div g 4.0
          k = mul f j
          _ = reduce_sum[ axes=(0,) ] k
          _ = log j
          l = mul i j
          m = add_any h l
      in (m,) }
    """
    print("Expected:", expected)
    print("Result:", jaxpr)
    #self.assertMultiLineStrippedEqual(expected, str(jaxpr))  # no jaxpr test

    self.assertIn('div', str(jaxpr))
    self.assertIn('log', str(jaxpr))  # assumes no DCE
    self.assertAllClose(jax.value_and_grad(lambda x: np.sum(f(x)))(x),
                        jax.value_and_grad(lambda x: np.sum(finv(x)))(x),
                        check_dtypes=True)

  @jtu.skip_on_devices("cpu")
  @jtu.ignore_warning(message="Values that an @invertible function closes")
  def test_perf(self):
    N, M = (64, 10**6)
    def f(x):
      def iterate(i, x): return lax.log(lax.exp(x) + 1)*0.9
      for i in range(N):
        x = iterate(i, x)
      return x

    finv = jax.invertible(f)

    @partial(jax.jit)
    def v_and_g(x):
      return jax.value_and_grad(lambda x: np.sum(f(x)))(x)

    @partial(jax.jit)
    def v_and_g_by_inv(x):
      return jax.value_and_grad(lambda x: np.sum(finv(x)))(x)

    MMul = 1
    while True:
      try:
        xval = jnp.ones((M * MMul,))
        v_and_g_result = v_and_g(xval)
        MMul *= 2
      except Exception as E:  
        # Memory error
        normal_ad_mem_limit = MMul
        break

    MMul = 1
    while True:
      try:
        xval = jnp.ones((M * MMul,))
        v_and_g_result_by_inv = v_and_g_by_inv(xval)
        MMul *= 2
      except Exception as E:  
        # Memory error
        inv_ad_mem_limit = MMul
        break

    print(f"Normal AD: OOM @ M={normal_ad_mem_limit}")
    print(f"Inverse AD: OOM @ M={inv_ad_mem_limit}")

    """
    stats1 = jax.devices()[0].memory_stats()['peak_bytes_in_use']
    v_and_g_result_by_inv = v_and_g_by_inv(xval)
    stats2 = jax.devices()[0].memory_stats()['peak_bytes_in_use']
    v_and_g_result = v_and_g(xval)
    stats3 = jax.devices()[0].memory_stats()['peak_bytes_in_use']
    print("\n".join((str(s) for s in (stats1, stats2, stats3))))
    """

    #normal_jaxpr = jax.make_jaxpr(v_and_g)(xval)
    #inv_jaxpr = jax.make_jaxpr(v_and_g_by_inv)(xval)

    xval = jnp.ones((M,))
    inv_time = timefunc(v_and_g_by_inv, xval)
    normal_time = timefunc(v_and_g, xval)

    print(f"Normal AD time: {normal_time}")
    print(f"Inverse AD time: {inv_time}")

    v_and_g_result = v_and_g(xval)
    v_and_g_result_by_inv = v_and_g_by_inv(xval)

    self.assertAllClose(v_and_g_result,
                        v_and_g_result_by_inv,
                        check_dtypes=True)

  def test_invertible_blocks(self):
    # NB: This is the reversible ResNet block
    def mk_reversible_block(f, g):
      @jax.custom_ivjp
      def rev_block(x1, x2):
        y1 = f(x2) + x1
        y2 = g(y1) + x2
        return y1, y2

      @rev_block.defivjp
      def rev_block_ivjp(xs, ys, dys):
        (y1, y2) = ys
        (dy1, dy2) = dys

        dgo, dx2 = dy2, dy2
        go, gvjp = jax.vjp(g, y1)
        dy1 += gvjp(dgo)[0]
        del gvjp
        x2 = y2 - go

        dfo, dx1 = dy1, dy1
        fo, fvjp = jax.vjp(f, x2)
        dx2 += fvjp(dfo)[0]
        del fvjp
        x1 = y1 - fo

        return (x1, x2), (dx1, dx2)

      return rev_block

    rev_block = mk_reversible_block(jnp.sin, jnp.cos)

    def g(x1, x2):
      for i in range(2):
        x1, x2 = rev_block(x1, x2)
      return x1, x2

    def reduce(f, x1, x2):
      y1, y2 = f(x1, x2)
      return np.sum(y1) + np.sum(y2)

    x = np.ones((1,))
    # FIXME: This breaks when argnums is left as default (i.e. 0), because JVP prunes
    #        zero tangents from call primitives.
    self.assertAllClose(jax.value_and_grad(partial(reduce, jax.invertible(g)), argnums=(0, 1))(x, x + 2),
                        jax.value_and_grad(partial(reduce, g), argnums=(0, 1))(x, x + 2),
                        check_dtypes=True)

  def test_invertible_partial_diff(self):
    # Check that we don't have to differentiate with respect to inputs
    # of the invertible function.
    def f(x, y):
      return lax.mul(lax.mul(lax.exp(x), 4.), x), lax.add(y, 4.)

    finv = jax.invertible(f)
    o = np.ones((5,))
    self.assertAllClose(jax.value_and_grad(lambda x: np.sum(f(x, o)[0]))(o),
                        jax.value_and_grad(lambda x: np.sum(finv(x, o)[0]))(o),
                        check_dtypes=True)

  def test_invertible_pytree(self):
    def f(x, y):
      return lax.add(lax.mul(lax.exp(x[0]), x[1]), y)

    finv = jax.invertible(f)
    o = np.ones((5,))
    self.assertAllClose(jax.value_and_grad(lambda x: np.sum(f((x, x), x)[0]))(o),
                        jax.value_and_grad(lambda x: np.sum(finv((x, x), x)[0]))(o),
                        check_dtypes=True)

def timefunc(f, *args, N=40):
    tic = time()
    for i in range(N):
      _ = f(*args)
    avg_runtime = (time() - tic) / N
    return avg_runtime

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
