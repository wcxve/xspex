import os
import pathlib
import re
import subprocess


def get_compiler():
    compiler = os.getenv('CXX')
    if compiler is not None:
        return compiler

    # We could try and be clever here, but for now just fall through
    # to g++ and assume that anyone using clang has to set CXX.  I do
    # not build on macOS so it's not a problem for me^(TM).
    #
    compiler = 'g++'
    return compiler


def compile_code(HEADAS):
    """Compile the code."""

    basename = 'report_xspec_version'
    helpers = pathlib.Path('helpers')

    compiler = get_compiler()
    args = [
        compiler,
        str(helpers / f'{basename}.cxx'),
        '-o',
        str(helpers / basename),
        f'-I{HEADAS}/include',
        f'-L{HEADAS}/lib',
        '-lXSUtil',
    ]

    subprocess.run(args, check=True)
    return helpers / basename


def get_xspec_macros(HEADAS):
    """Return the macro definitions which define the Xspec version.

    Parameters
    ----------
    HEADAS : pathlib.Path
        The path to the HEADAS location.

    Returns
    -------
    xspec_version, macros : str, list
        The Xspec version, including the patch level, and then the
        macro definitions to pass to the compiled code.

    """

    code = compile_code(HEADAS)
    command = subprocess.run([str(code)], check=True, stdout=subprocess.PIPE)

    xspec_version = command.stdout.decode().strip()
    print(f"Building against Xspec: '{xspec_version}'")

    # split the Xspec version
    toks = xspec_version.split('.')
    assert len(toks) == 3, xspec_version
    xspec_major = toks[0]
    xspec_minor = toks[1]

    match = re.match(r'^(\d+)(.*)$', toks[2])
    xspec_micro = match[1]
    xspec_patch = None if match[2] == '' else match[2]

    macros = [
        ('BUILD_Xspec', xspec_version),
        ('BUILD_Xspec_MAJOR', xspec_major),
        ('BUILD_Xspec_MINOR', xspec_minor),
        ('BUILD_Xspec_MICRO', xspec_micro),
    ]

    if xspec_patch is not None:
        macros.append(('BUILD_Xspec_PATCH', xspec_patch))

    return xspec_version, macros
