# xspex

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xspex?color=blue&logo=Python&logoColor=white&style=for-the-badge)](https://pypi.org/project/xspex)
[![PyPI - Version](https://img.shields.io/pypi/v/xspex?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/xspex)
[![License: GPL v3](https://img.shields.io/github/license/wcxve/xspex?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)<br>
[![Coverage Status](https://img.shields.io/codecov/c/github/wcxve/xspex?logo=Codecov&logoColor=white&style=for-the-badge)](https://app.codecov.io/github/wcxve/xspex)

JAX interface for XSPEC spectral models.

## Installation

Note: ``HEASoft`` & ``XSPEC v12.12.1+`` are required to be installed on your
system.
You can download it from [here](https://heasarc.gsfc.nasa.gov/lheasoft/).

```shell
pip install xspex
```

## Examples

### Basic Usage

```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import jax
import jax.numpy as jnp
import numpy as np
import xspex as xx

# Double precision is required for XSPEC models
jax.config.update('jax_enable_x64', True)

# Get APEC model function
fn, info = xx.get_model('apec')

# Define parameters and energy grid
params = jnp.array([1.0, 1.0, 0.0])
egrid = jnp.linspace(0.1, 0.2, 6)

# Evaluate the model function
value = fn(params, egrid)
print(value)
# output:
# [1.29398149 0.43733974 0.11998129 0.08725635 0.07757024]
```

### JAX Transformations

``xspex`` provides JAX automatic differentiation support for XSPEC models through finite difference approximation. This allows seamless integration with JAX's transformations like ``grad``, ``jacfwd``, ``jacrev``, etc.

#### Computing Gradients

```python
# Get gradient function with respect to parameters
grad_fn = jax.grad(lambda p, e: jnp.sum(fn(p, e)))

# Compute gradient, note that the abundance and redshift are fixed by default
grad = grad_fn(params, egrid)
print(grad)
# output:
# [-3.1665168  0.         0.       ]
```

#### Computing Jacobian

```python
# Get Jacobian function
jac_fn = jax.jacfwd(lambda p, e: fn(p, e))  # or jax.jacrev, jax.jacobian

# Compute Jacobian matrix
jacobian = jac_fn(params, egrid)
print(jacobian)
# output:
# [[-2.01717805 -0.         -0.        ]
#  [-1.05626962 -0.         -0.        ]
#  [-0.03252301 -0.         -0.        ]
#  [-0.02018553 -0.         -0.        ]
#  [-0.0403606  -0.         -0.        ]]
```

#### Vectorization with `vmap`

```python
# Create multiple parameter sets
param_sets = jnp.array([
    [0.5, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [2.0, 1.0, 0.0],
])

# Vectorize the function
vmapped_fn = jax.vmap(fn, in_axes=(0, None))
results = vmapped_fn(param_sets, egrid)
print(results)
# output:
# [[0.52477309 0.56379027 0.13421626 0.11663016 0.17570166]
#  [1.29398149 0.43733974 0.11998129 0.08725635 0.07757024]
#  [0.18578006 0.10865312 0.08878279 0.07398064 0.06353859]]
```

#### Parallel evaluation with `pmap`

```python
# Replicate parameters across devices
param_sets = jnp.array([
    [1.0, 1.0, 0.0],
    [2.0, 1.0, 0.0],
])

pmapped_fn = jax.pmap(fn, in_axes=(0, None))
results = pmapped_fn(param_sets, egrid)
print(results)
# output:
# [[1.29398149 0.43733974 0.11998129 0.08725635 0.07757024]
#  [0.18578006 0.10865312 0.08878279 0.07398064 0.06353859]]
```

### Custom Finite Difference Automatic Differentiation

```python
# Create model with custom finite difference settings
fn, info = xx.get_model('powerlaw')
fn2 = xx.define_fdjvp(  # see the docstring for more details
    fn,
    info,
    delta=1e-6,           # Custom step size (relative to parameter value)
    method='central',     # 'central' or 'forward' finite differences
    fixed=None            # Optional: specify which parameters to keep fixed
)
```
