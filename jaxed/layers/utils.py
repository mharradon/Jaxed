import jax

from flax.core.scope import LazyRng

def make_rng(parent_mod, name):
    # Circumvent Flax safety checks
    # This breaks JIT
    """
    scope = parent_mod.scope
    rng = scope.rngs[name]
    scope.rng_counters[name] += 1
    return LazyRng.create(rng, scope.rng_counters[name]).as_jax_rng()
    """
    # Awful jank
    return jax.random.key(0)

def lower_submodule_to_function(parent_mod, target_mod_ref, x):
    # Take a target module and return (f, p) such that:
    # 1) A pure JAX function f(x, p) describing the __call__ operation on target mod
    # 2) A set of params on parent_mod that can be passed into f to replicate the original calculation

    target_mod = getattr(parent_mod, target_mod_ref)

    def expand_init(rng, *args, **kwargs):
        if len(parent_mod.scope.rngs) > 0:
            if not isinstance(rng, dict):
                rng = {'params': rng}
            rng.update({k: make_rng(parent_mod, k) for k in parent_mod.scope.rngs.keys() if k != 'params'})
        return target_mod.init(rng, *args, **kwargs)

    params = parent_mod.param(target_mod_ref + '_shadow',
                              expand_init,
                              *x)

    def func(args, params):
        if len(parent_mod.scope.rngs) > 0:
            rngs = {k: make_rng(parent_mod, k) for k in parent_mod.scope.rngs.keys() if k != 'params'}
        else:
            rngs = None
        return target_mod.apply(params, *args, rngs=rngs)

    return func, params
