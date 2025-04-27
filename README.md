# xspex

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xspex?color=blue&logo=Python&logoColor=white&style=for-the-badge)](https://pypi.org/project/xspex)
[![PyPI - Version](https://img.shields.io/pypi/v/xspex?color=blue&logo=PyPI&logoColor=white&style=for-the-badge)](https://pypi.org/project/xspex)
[![License: GPL v3](https://img.shields.io/github/license/wcxve/xspex?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)<br>
[![Coverage Status](https://img.shields.io/codecov/c/github/wcxve/xspex?logo=Codecov&logoColor=white&style=for-the-badge)](https://app.codecov.io/github/wcxve/xspex)

Access Xspec models and corresponding JAX/XLA ops, based on [xspec_models_cxc](https://github.com/cxcsds/xspec-models-cxc/) and [extending-jax](https://github.com/dfm/extending-jax).

## Installation

Note: ``HEASoft`` & ``Xspec v12.12.1+`` are required to be installed on your system.
You can download it from [here](https://heasarc.gsfc.nasa.gov/lheasoft/).

```bash
pip install xspex
```

## Example

```python
import jax
import jax.numpy as jnp
import numpy as np
import xspex

# For accuracy, it is recommended to enable double precision
jax.config.update('jax_enable_x64', True)

# Get APEC model primitive, whose JVP rule is defined by finite difference
apec, info = xspex.get_primitive('apec')

# Evaluate the model via JAX primitive
value = apec(
    params=jnp.array([1.0, 1.0, 0.0]),
    egrid=jnp.geomspace(0.1, 0.2, 6),
    spec_num=1,
)
print(value)  # [1.27358561 0.37946811 0.2477116  0.1071355  0.10049102]

# Evaluate the model function
value2 = xspex.apec(
    pars=np.array([1.0, 1.0, 0.0]),
    energies=np.geomspace(0.1, 0.2, 6),
)
print(value2)  # [1.27358561 0.37946811 0.2477116  0.1071355  0.10049102]
```
