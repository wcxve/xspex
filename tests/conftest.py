from __future__ import annotations

import chex
import jax

# Use 2 CPU devices in JAX
chex.set_n_cpu_devices(2)

# Use double precision in JAX
jax.config.update('jax_enable_x64', True)

XSPEC_ABUND_TABLES = (
    'angr',
    'aspl',
    'feld',
    'aneb',
    'grsa',
    'wilm',
    'lodd',
    'lpgp',
    'lpgs',
)

XSPEC_XSECT_TABLES = (
    'bcmc',
    'obcm',
    'vern',
)

# models with occasional failures
MODELS_XFAIL = (
    'bwcycl',
    'grbjet',
    'ismdust',
    'olivineabs',
    'xion',
)
