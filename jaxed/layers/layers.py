import jax
import flax.linen as nn

import jaxed.internals

from .utils import lower_submodule_to_function

class Invertible(nn.Module):
    mod: nn.Module
    @nn.compact
    def __call__(self, *args):
        callfunc, params = lower_submodule_to_function(self, 'mod', args)
        callfunc = jax.invertible(callfunc)
        return callfunc(args, params)
