import glob
import os
import pathlib
import platform
import sys
import sysconfig

from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

sys.path.append(os.path.dirname(__file__))
from helpers import template
from helpers.identify_xspec import get_xspec_macros

__version__ = '0.0.3'

# Check HEASARC is set up. The following does not provide a useful
# message from 'pip install' so how do we make it more meaningful?
#
HEADAS = os.getenv('HEADAS')
if HEADAS is None:
    sys.stderr.write('ERROR: unable to find HEADAS environment variable.\n')
    sys.exit(1)

HEADAS = pathlib.Path(HEADAS)

modelfile = HEADAS / '../spectral/manager/model.dat'
modelfile = modelfile.resolve()

out_dir = pathlib.Path('src')
out_dir.mkdir(exist_ok=True)

# It would be nice to query for this from the system,
# such as with pkg_config. We can try and find the versions
# directly instead.
# xspec_libs = ['XSFunctions', 'XSUtil', 'XS', 'hdsp_6.29',
#               'cfitsio', 'CCfits_2.6', 'wcs-7.3.1']

# There's some attempt to be platform independent, but
# is it worth it?
libdir = HEADAS / sysconfig.get_config_var('platlibdir')
if platform.system() == 'Darwin':
    suffix = '.dylib'
else:
    suffix = sysconfig.get_config_var('SHLIB_SUFFIX')


# The tricky thing is that we have XSFunctions, XSUtil, and XS as
# arguments. So we cannot just look for XS*, as that will match
# multiple libraries. We also don't want to include all matches to XS
# as there are a number of matches we do not need.
def match(name):
    # Would it make sense to take the lib prefix from sysconfig?
    head = f'lib{name}{suffix}'
    ms = glob.glob(str(libdir / head))
    if len(ms) == 1:
        return name

    head = f'lib{name}_*{suffix}'
    ms = glob.glob(str(libdir / head))
    if len(ms) == 1:
        return pathlib.Path(ms[0]).stem[3:]

    head = f'lib{name}-*{suffix}'
    ms = glob.glob(str(libdir / head))
    if len(ms) == 1:
        return pathlib.Path(ms[0]).stem[3:]

    raise OSError(f'Unable to find a match for lib{name}*{suffix} in {libdir}')


xspec_libs = []
for libname in ['XSFunctions', 'XSUtil', 'XS', 'hdsp',
                'cfitsio', 'CCfits', 'wcs']:
    # Note: not all names are versioned
    xspec_libs.append(match(libname))


xspec_version, xspec_macros = get_xspec_macros(HEADAS)


# Create the code now we believe we have the Xspec installation
# sorted out.
info = template.apply(modelfile, xspec_version, out_dir)


# Note: we need access to the src/include directory - can we just
# hard-code this path or access it via some setuptools method?
include_dir = out_dir / 'include'
if not include_dir.is_dir():
    sys.stderr.write(f'ERROR: unable to find {include_dir}/')
    sys.exit(1)

macros = [('VERSION_INFO', __version__)] + xspec_macros


ext_modules = [
    Pybind11Extension(
        'xspex._compiled',
        [str(info['outfile'])],
        cxx_std=11,
        extra_compile_args=['-O3'],
        include_dirs=[str(include_dir), str(HEADAS / 'include')],
        library_dirs=[str(HEADAS / 'lib')],
        runtime_library_dirs=[str(HEADAS / 'lib')],
        libraries=xspec_libs,
        define_macros=macros
    )
]

setup(
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
