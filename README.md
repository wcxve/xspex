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

```bash
pip install xspex
```

## Example

```python
import jax
import jax.numpy as jnp
import numpy as np
import xspex as xx

# Double precision is required for XSPEC models
jax.config.update('jax_enable_x64', True)

# Get APEC model function
fn, info = xx.get_model('apec')

# Evaluate the model via JAX primitive
value = fn(
    params=jnp.array([1.0, 1.0, 0.0]),
    egrid=jnp.geomspace(0.1, 0.2, 6),
    spec_num=1,
)
print(value)  # [1.27358561 0.37946811 0.2477116  0.1071355  0.10049102]
```
