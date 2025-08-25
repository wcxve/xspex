from __future__ import annotations

import chex
import jax

chex.set_n_cpu_devices(4)

# Use double precision in JAX
jax.config.update('jax_enable_x64', True)

# # Set NEI version used in XSPEC
# Xset.addModelString('NEIVERS', '3.1.2')
# version = xx.xspec_version().split('.')
# major = int(version[0])
# minor = int(version[1])
# if major > 12 or (major == 12 and minor >= 15):
#     xx.set_mstr('NEIVERS', '3.1.2')
