<h1 align="center">
Jaxed
</h1>
<h2 align="center">
Reversible Automatic Differentiation in Jax
</h2>

[![docs](https://github.com/mharradon/Jaxed/actions/workflows/docs.yml/badge.svg)](https://mharradon.github.io/Jaxed/)

Reversible Automatic Differentiation[^1] (RAD) is a modification of the standard Reverse-Mode Automatic Differentiation (AD) algorithm used in deep learning designed to greatly reduce GPU memory requirements.

Typically in AD the activations of each layer are saved to be re-used in the calculation of a jacobian vector product (JVP) during the backwards pass. This is often the dominant contributors to memory usage during training. The idea of RAD is to recalculate activations on-the-fly by using invertible functions for all or part of a deep neural network architecture. During the backwards pass the activations can then by calculated one-by-one as the algorithm proceeds backwards through the network.

Applying RAD to a DNN requires two things:

1. Implementations of inverse operations for the invertible functions employed, and
2. An alternate implementation of AD that uses these inverse operators to calculate JVPs in place of pre-computed activations

This package resurrects a JAX[^2] implementation of RAD as a core JAX interpreter and aims to collect a variety of invertible functions and Flax[^3] layers.

[^1]: Gomez, Aidan N., et al. "The reversible residual network: Backpropagation without storing activations." Advances in neural information processing systems 30 (2017).
[^2]: https://github.com/google/jax
[^3]: https://github.com/google/flax
