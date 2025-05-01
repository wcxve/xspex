import jax

import xspex as x

# We want to ensure we have a fixed abundance / cross section
# for the tests. If we didn't set them here then it would depend
# on the user's ~/.xspec/Xspec.init file
x.abundance('lodd')
x.cross_section('vern')

# Set nei version
version = x.version().split('.')
major = int(version[0])
minor = int(version[1])
if major > 12 or (major == 12 and minor >= 15):
    x.set_model_string('NEIVERS', '3.1.2')

# Set JAX float precision to double
jax.config.update('jax_enable_x64', True)
