# xspex

Access Xspec models and corresponding JAX/XLA ops, based on [xspec_models_cxc](https://github.com/cxcsds/xspec-models-cxc/) and [extending-jax](https://github.com/dfm/extending-jax).

## Installation

Note: `HEASoft` & `Xspec` are required to be installed on your system. You can download it from [here](https://heasarc.gsfc.nasa.gov/lheasoft/).

```bash
pip install xspex
```

## Example

```python
import jax.numpy as jnp
import xspex

# Get APEC model primitive, whose JVP rule is defined by finite difference 
apec, info = xspex.get_primitive('apec')

# Evaluate the model
value = apec(
    params=jnp.array([1.0, 1.0, 0.0]),
    egrid=jnp.geomspace(0.1, 0.2, 6),
    spec_num=1,
)
print(value)  # [1.2735856  0.37946814 0.24771158 0.10713547 0.10049101]
```
