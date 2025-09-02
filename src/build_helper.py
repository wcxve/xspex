import argparse
import glob
import os
import sys
from pathlib import Path
from sysconfig import get_config_var

sys.path.insert(0, (Path(__file__).parent / 'xspex').resolve(True).as_posix())
from _xspec import model_parser, types


def get_headas_include_dir() -> str:
    """Get the HEADAS include directory."""
    include_dir = Path(model_parser.get_headas_env()) / 'include'
    include_dir = include_dir.resolve()
    if not include_dir.exists():
        raise OSError('HEADAS include directory not found')
    return include_dir.as_posix()


def get_headas_lib_dir() -> str:
    """Get the HEADAS library directory."""
    HEADAS = Path(model_parser.get_headas_env())
    lib_dir = HEADAS / 'lib'
    lib_dir = lib_dir.resolve()
    if not lib_dir.exists():
        lib_dir = HEADAS / get_config_var('platlibdir')
        lib_dir = lib_dir.resolve()
        if not lib_dir.exists():
            raise OSError('HEADAS library directory not found')
    return lib_dir.as_posix()


def find_headas_library_path(name: str) -> str:
    """Find HEADAS library path."""
    lib_dir = get_headas_lib_dir()
    version_patterns = ['', '_*', '-*', '*']
    prefix_patterns = ['lib', '']
    suffix_patterns = ['.so', '.a']
    if get_config_var('SHLIB_SUFFIX') not in suffix_patterns:
        suffix_patterns.insert(0, get_config_var('SHLIB_SUFFIX'))
    if get_config_var('WITH_DYLD'):
        suffix_patterns.insert(0, '.dylib')
    for version in version_patterns:
        for prefix in prefix_patterns:
            for suffix in suffix_patterns:
                p = f'{prefix}{name}{version}{suffix}'
                files = glob.glob(os.path.join(lib_dir, p))
                if len(files):
                    return sorted(files)[0]

    raise OSError(f'Failed to find library {name} in {lib_dir}')


def print_xla_include_dir():
    """Print the XLA FFI include directory."""
    from jax import ffi

    print(ffi.include_dir())


def print_headas_include_dir():
    """Print the HEADAS include directory."""
    print(get_headas_include_dir())


def print_headas_lib_dir():
    """Print the HEADAS library directory."""
    print(get_headas_lib_dir())


def print_headas_library_path(name: str):
    """Print the library path."""
    print(find_headas_library_path(name))


XSPEC_MODELS = model_parser.get_models_info(parse_desc_and_link=False)


def print_xspec_models():
    """Print the models."""
    for model in XSPEC_MODELS.values():
        if model.type != types.XspecModelType.Con:
            print(f'ENTRY({model.name}, {model.n_params})', end=' ')


def print_xspec_con_models():
    """Print the convolution models."""
    for model in XSPEC_MODELS.values():
        if model.type == types.XspecModelType.Con:
            print(f'ENTRY({model.name}, {model.n_params})', end=' ')


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description='Helper functions for XSPEX build configuration'
    )
    parser.add_argument('function', help='Function to call')
    parser.add_argument('args', nargs='*', help='Arguments for the function')

    args = parser.parse_args()

    allowed_functions = {
        'print_xla_include_dir': print_xla_include_dir,
        'print_headas_include_dir': print_headas_include_dir,
        'print_headas_lib_dir': print_headas_lib_dir,
        'print_headas_library_path': print_headas_library_path,
        'print_xspec_models': print_xspec_models,
        'print_xspec_con_models': print_xspec_con_models,
    }

    if args.function not in allowed_functions:
        raise ValueError(f'Unknown function: {args.function}')
    elif args.function == 'print_headas_library_path':
        if len(args.args) != 1:
            raise ValueError(
                'print_headas_library_path requires 1 argument: name'
            )
        print_headas_library_path(args.args[0])
    else:
        allowed_functions[args.function]()


if __name__ == '__main__':
    main()
