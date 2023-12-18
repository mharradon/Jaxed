def lower_submodule_to_function(parent_mod, target_mod_ref, x):
    # Take a target module and return (f, p) such that:
    # 1) A pure JAX function f(x, p) describing the __call__ operation on target mod
    # 2) A set of params on parent_mod that can be passed into f to replicate the original calculation

    target_mod = getattr(parent_mod, target_mod_ref)

    params = parent_mod.param(target_mod_ref + '_shadow',
                              target_mod.init,
                              *x)

    def func(args, params):
      return target_mod.apply(params, *args)

    return func, params
