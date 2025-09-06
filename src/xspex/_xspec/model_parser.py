from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from .types import (
    XspecFuncType,
    XspecModel,
    XspecModelType,
    XspecParam,
    XspecParamType,
)

if TYPE_CHECKING:
    from typing import TextIO, TypedDict

    class ModelDescLink(TypedDict):
        desc: str
        link: str


_P = {
    # start of line with optional whitespace characters
    '^': r'^\s*',
    # end of line with optional whitespace characters
    '$': r'\s*$',
    # whitespace separator
    's': r'\s+',
    # non-whitespace characters
    'S': r'(?P<{}>\S+)',
    # any string
    'a': r'(?P<{}>(?=\S).*\S)',
    # model type
    'm': r'(?P<mtype>add|mul|con|mix|acn|amx)',
    # boolean flag, 0 or 1
    'b': r'(?P<{}>[01])',
    # positive integer numbers
    'i': r'(?P<{}>[0-9]+)',
    # floating-point numbers, adapted from
    # Regular Expressions Cookbook, 2nd Edition, Chapter 6.10
    'f': r'(?P<{}>[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)',
    # floating-point numbers, no capture
    'F': r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?',
    # unit, in quotes or not
    'u': r'(?P<unit>"[^"]*"|[^"\s]+)',
    # unit, no capture
    'U': r'(?:"[^"]*"|\S+)',
    # periodic flag, starts with p or P
    'p': r'(?:\s+(?P<periodic>[pP]\S*))?',
}

# pattern for model definition line
_MODEL_PATTERN = re.compile(
    pattern=(
        '{^}'
        '{S}'  # model name
        '{s}{i}'  # number of parameters
        '{s}{f}'  # minimum valid energy
        '{s}{f}'  # maximum valid energy
        '{s}{S}'  # model function name
        '{s}{m}'  # model type
        '{s}{b}'  # flag of whether the model calculates errors
        '(?:'  # optional model settings
        '{s}{b}'  # flag of whether the model is data dependent
        '(?:'  # initialization string and extra string
        '{s}{S}'  # initialization string
        '(?:{s}{a})?'  # extra string is not used in XSPEC
        ')?'
        ')?'
        '{$}'
    )
    .format_map(_P)
    .format(
        'mname',
        'npars',
        'emin',
        'emax',
        'func',
        'calc_errors',
        'data_depend',
        'init_string',
        'extra_string',
    )
)

# pattern for default parameter line
_DEFAULT_PARAMETER_PATTERN = re.compile(
    pattern=(
        '{^}'
        '{S}'  # parameter name
        '{s}{u}'  # parameter unit
        '{s}{f}'  # default value
        '{s}{f}'  # hard minimum
        '{s}{f}'  # soft minimum
        '{s}{f}'  # soft maximum
        '{s}{f}'  # hard maximum
        '{s}{f}'  # delta
        '{p}'  # periodic flag
        '{$}'
    )
    .format_map(_P)
    .format('pname', 'default', 'min', 'bot', 'top', 'max', 'delta')
)

# pattern for switch parameter line
# XSPEC (12.12.1 to 12.15.0) model.dat file has some switch parameters with
# unit, value range, and delta specified, which is illegal according to
# XSPEC manual (Appendix C), so we don't capture these parts
_SWITCH_PARAMETER_PATTERN = re.compile(
    pattern=(
        '{^}'
        r'\${S}'  # switch parameter name
        '(?:{s}{U})?'  # unit, optional
        '{s}{f}'  # default value
        '(?:{s}{F}){{{{0,5}}}}'  # value range and delta, optional
        '{$}'
    )
    .format_map(_P)
    .format('pname', 'default')
)

# pattern for scale parameter line
# XSPEC (12.12.1 to 12.15.0) model.dat file has some scale parameters with
# value range and delta specified, which is illegal according to XSPEC manual
# (Appendix C), so we don't capture the value range and delta part
_SCALE_PARAMETER_PATTERN = re.compile(
    pattern=(
        '{^}'
        r'\*{S}'  # scale parameter name
        '{s}{u}'  # unit
        '{s}{f}'  # default value
        '(?:{s}{F}){{{{0,5}}}}'  # value range and delta, optional
        '{$}'
    )
    .format_map(_P)
    .format('pname', 'default')
)


def get_headas_env() -> str:
    """Get the HEADAS environment variable."""
    if 'HEADAS' not in os.environ:
        raise OSError('HEADAS environment variable not found')
    return os.environ['HEADAS']


def get_spectral_path() -> str:
    """Get XSPEC spectral path."""
    # In the HEASoft Conda distribution, the spectral directory is located
    # under "$HEADAS/spectral". However, in the XSPEC source code (v12.15.0b
    # and earlier, XSFunctions/Utilities/xsFortran.cxx, line 198), the spectral
    # path is hardcoded as "$HEADAS/../spectral". This mismatch can lead to
    # file access errors during runtime. The solution is to manually create a
    # symbolic link, from "$HEADAS/../spectral" to "$HEADAS/spectral".
    HEADAS = get_headas_env()
    spectral_path = Path(HEADAS).parent / 'spectral'
    spectral_path = spectral_path.resolve()
    if not spectral_path.exists():
        alt_path = Path(HEADAS) / 'spectral'
        alt_path = alt_path.resolve()
        if alt_path.exists() and alt_path.is_dir():
            spectral_path.symlink_to(target=alt_path, target_is_directory=True)
        else:
            raise OSError('XSPEC spectral directory not found')
    return spectral_path.as_posix()


def get_models_desc_and_link() -> dict[str, ModelDescLink]:
    """Parse XSPEC model description and link."""
    # import bs4 here to avoid import error when building the package
    from bs4 import BeautifulSoup

    spectral_path = Path(get_spectral_path())

    url = 'https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodel{}.html'
    html_path = spectral_path / 'help' / 'html'
    model_info: dict[str, ModelDescLink] = {}
    for mtype in [
        'Additive',
        'Multiplicative',
        'Convolution',
        'Pileup',
        'Mixing',
    ]:
        with open(html_path / f'{mtype}.html', encoding='utf-8') as f:
            s = BeautifulSoup(f.read(), 'html.parser')

        for a in s.find_all('ul', class_='ChildLinks')[0].find_all('a'):  # type: ignore
            text = str(a.text)  # type: ignore

            # there is no ':' in agnslim model desc
            if mtype == 'Additive' and text.startswith('agnslim, AGN'):
                text = text.replace('agnslim, AGN', 'agnslim: AGN')

            models, desc = text.split(':')
            models_list = [m.strip() for m in models.split(',')]
            desc = desc.strip()
            desc = re.sub(r'\s+', ' ', desc)
            if desc and desc[0].islower():
                desc = desc[0].upper() + desc[1:]
            if not desc.endswith('.'):
                desc += '.'
            link = url.format(models_list[0].capitalize())
            for m in models_list:
                model_info[m.casefold()] = {'desc': desc, 'link': link}

    # There are typos in some models' names
    if 'bvvcie' not in model_info and 'bbvcie' in model_info:
        model_info['bvvcie'] = model_info.pop('bbvcie')
    if 'bvwdem' not in model_info and 'bwwdem' in model_info:
        model_info['bvwdem'] = model_info.pop('bwwdem')

    # Some models are not added in XSPEC manual yet
    if 'bfeklor' not in model_info:
        model_info['bfeklor'] = {
            'desc': 'velocity broaden the Fe K alpha complex',
            'link': url.format('Bfeklor'),
        }
        model_info['zbfeklor'] = {
            'desc': 'velocity broaden the Fe K alpha complex',
            'link': url.format('Bfeklor'),
        }

    return model_info


def get_model_file() -> str:
    """Get XSPEC model.dat file."""
    spectral_path = Path(get_spectral_path())
    modelfile = spectral_path / 'manager' / 'model.dat'
    return modelfile.resolve(strict=True).as_posix()


def get_models_info(
    model_file: str | None = None,
    parse_desc_and_link: bool = True,
) -> dict[str, XspecModel]:
    """Get models information from XSPEC model.dat file.

    The parsing logic follows the XSPEC source code, see
        - XSFunction/Utilities/XSModelFunction.cxx, updateComponentList

    Parameters
    ----------
    model_file : str
        The path to the XSPEC model.dat file. If None, automatically find the
        file under HEADAS spectral directory.
    parse_desc_and_link : bool, optional
        Whether to parse description and link of the models. Default is True.

    Returns
    -------
    models : dict
        The parsed models. The keys are the model names, and the values are the
        parsed models.
    """
    if parse_desc_and_link:
        model_desc_and_link = get_models_desc_and_link()
    else:
        model_desc_and_link = None

    # The keys of models are formatted model names, i.e., the model names in
    # the XSPEC manual, which are lowercased and without underscores.
    # The values are the parsed models. The model has attr name, which is the
    # name in model.dat, and the corresponding C function in XSPEC is C_<name>.
    models: dict[str, XspecModel] = {}

    def format_name(name: str) -> str:
        """Format model name in model.dat to match that in the XSPEC manual."""
        name = name.casefold()
        name = name.replace('_', '')
        if name.endswith('gaussian'):
            name = name.replace('gaussian', 'gauss')
        return name

    def parse_next_model(f: TextIO) -> bool:
        """Parse a model line from XSPEC model.dat file."""
        # read until a non-empty line is found, or EOF is reached
        while mline := f.readline():
            mline = mline.strip()
            if mline:
                break

        # EOF, return None
        if mline == '':
            return False

        match = _MODEL_PATTERN.match(mline)
        if match is None:
            raise ValueError(f'invalid model definition: {mline}')

        mtype = XspecModelType.from_str(match.group('mtype'))
        mname = match.group('mname')
        func, func_type, eps = XspecFuncType.get_func_meta(match.group('func'))
        n_params = int(match.group('npars'))
        emin = float(match.group('emin'))
        emax = float(match.group('emax'))
        calc_errors = match.group('calc_errors') not in (None, '0')
        data_depend = match.group('data_depend') not in (None, '0')
        init_string = match.group('init_string') or ''
        extra_string = match.group('extra_string') or ''
        if extra_string:
            warnings.warn(f'extra string is not supported: {mline}')

        # parse parameters
        parameters: list[XspecParam] = []
        for i in range(n_params):
            if not (pline := f.readline().strip()):
                n_miss = n_params - i - 1
                raise ValueError(f'{mname} model missing {n_miss} parameters')

            pline = _fix_parameter_definition(mname, pline)

            try:
                p = _parse_parameter(pline)
            except ValueError as e:
                raise ValueError(
                    f'{mname} model has invalid parameter definition:\n{pline}'
                ) from e

            parameters.append(p)

        mname_formatted = format_name(mname)

        if model_desc_and_link is None:
            desc = ''
            link = ''
        else:
            desc_and_link = model_desc_and_link[mname_formatted]
            desc = desc_and_link['desc']
            link = desc_and_link['link']

        models[mname_formatted] = XspecModel(
            type=mtype,
            name=mname,
            func=func,
            func_type=func_type,
            eps=eps,
            desc=desc,
            link=link,
            emin=emin,
            emax=emax,
            parameters=tuple(parameters),
            calc_errors=calc_errors,
            data_depend=data_depend,
            init_string=init_string,
        )

        return True

    with open(model_file or get_model_file()) as f:
        while parse_next_model(f):
            pass

    return models


def _fix_parameter_definition(mname: str, pline: str) -> str:
    """Fix parameter definition.

    Some models have parameters with default value out of allowed range, so we
    fix them here. This is a temporary solution, and should be removed in the
    future. XSPEC v12.15.0 is found to have this issue.
    """
    if mname.endswith('vlorentz'):
        if 'Width' in pline:
            pline = pline.replace('100.', '10.')
    elif mname.endswith('vvoigt'):
        if 'Sigma' in pline:
            pline = pline.replace('100.', '10.')
        elif 'Gamma' in pline:
            pline = pline.replace('100.', '10.')
    elif mname.endswith('vlorabs'):
        if 'Width' in pline:
            pline = pline.replace('100.', '10.')
    elif mname.endswith('vvoigtabs'):
        if 'Width' in pline:
            pline = pline.replace('100.', '10.')
        elif 'Sigma' in pline:
            pline = pline.replace('100.', '10.')

    return pline


def _parse_parameter(line: str) -> XspecParam:
    """Parse a parameter line from XSPEC model.dat file.

    The parsing logic follows the XSPEC source code, see

        - XSModel/Parameter/ParamCreator.cxx
        - XSModel/Parameter/ModParam.cxx
        - XSModel/Parameter/SwitchParam.cxx
        - XSModel/Parameter/ScaleParam.cxx

    Parameters
    ----------
    line : str
        The parameter line to parse.

    Returns
    -------
    parameter : XSPECParameter
        The parsed parameter.
    """
    match line[0]:
        case '$':
            return _parse_switch_parameter(line)
        case '*':
            return _parse_scale_parameter(line)
        case _:
            return _parse_default_parameter(line)


def _parse_default_parameter(line: str) -> XspecParam:
    """Parse a default parameter line from XSPEC model.dat file."""
    match = _DEFAULT_PARAMETER_PATTERN.match(line)
    if match is None:
        raise ValueError(f'invalid parameter definition: {line}')

    name = _sanitize_name(match.group('pname'))
    unit = match.group('unit').replace('"', '').strip()
    default = float(match.group('default'))
    min = float(match.group('min'))
    bot = float(match.group('bot'))
    top = float(match.group('top'))
    max = float(match.group('max'))
    periodic = match.group('periodic') is not None

    if not (min <= bot <= top <= max):
        raise ValueError(f'invalid parameter range: {line}')

    if not (min <= default <= max):
        raise ValueError(f'invalid default value: {line}')

    if periodic:
        raise ValueError(f'periodic parameter is not supported yet: {line}')

    delta = float(match.group('delta'))
    fixed = False
    if delta <= 0:
        delta = abs(delta)
        fixed = True

    return XspecParam(
        type=XspecParamType.Basic,
        name=name,
        unit=unit,
        default=default,
        min=min,
        bot=bot,
        top=top,
        max=max,
        delta=delta,
        fixed=fixed,
        periodic=periodic,
    )


def _parse_switch_parameter(line: str) -> XspecParam:
    """Parse a switch parameter line from XSPEC model.dat file."""
    match = _SWITCH_PARAMETER_PATTERN.match(line)
    if match is None:
        raise ValueError(f'invalid switch parameter definition: {line}')

    name = _sanitize_name(match.group('pname'))

    # Switch parameter requires an integer, floating-point value will be
    # truncated.
    default = int(float(match.group('default')))

    return XspecParam(
        type=XspecParamType.Switch,
        name=name,
        unit='',
        default=default,
        min=None,
        bot=None,
        top=None,
        max=None,
        delta=None,
        fixed=True,
        periodic=False,
    )


def _parse_scale_parameter(line: str) -> XspecParam:
    """Parse a scale parameter line from XSPEC model.dat file."""
    match = _SCALE_PARAMETER_PATTERN.match(line)
    if match is None:
        raise ValueError(f'invalid scale parameter definition: {line}')

    name = _sanitize_name(match.group('pname'))
    unit = match.group('unit').replace('"', '').strip()
    default = float(match.group('default'))

    return XspecParam(
        type=XspecParamType.Scale,
        name=name,
        unit=unit,
        default=default,
        min=None,
        bot=None,
        top=None,
        max=None,
        delta=None,
        fixed=True,
        periodic=False,
    )


def _sanitize_name(name: str) -> str:
    """Sanitize parameter name to avoid Python keyword."""
    # import keyword
    # if keyword.iskeyword(name):
    #     return f'{name}_'
    return name
