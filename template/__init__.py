from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .primitive import (
        XspecConvPrimitive as XspecConvPrimitive,
        XspecPrimitive as XspecPrimitive,
    )

try:
    from . import _compiled
    from ._compiled import *  # noqa

    _compiled._init()
    __version__ = _compiled.__version__
    __INITIALIZED__ = True
except ImportError as ie:
    # Allow the actual error message to be reported if the user
    # has tweaked the log level, for instance with:
    #
    #   import logging; logging.basicConfig(level=logging.DEBUG)
    #
    logging.getLogger(__name__).warn('Unable to import compiled Xspec models')
    logging.getLogger(__name__).info(str(ie))

    __version__ = 'none'
    __INITIALIZED__ = False


class ModelType(Enum):
    """The various Xspec model types."""

    Add = auto()
    Mul = auto()
    Con = auto()


class LanguageStyle(Enum):
    """The various ways to define and call Xspec models."""

    CppStyle8 = auto()
    CStyle8 = auto()
    F77Style4 = auto()
    F77Style8 = auto()


class ParamType(Enum):
    """The Xspec parameter type."""

    Default = auto()
    Switch = auto()
    Scale = auto()
    Periodic = auto()


@dataclass
class XspecParameter:
    """An Xspec parameter."""

    paramtype: ParamType
    name: str
    default: float
    units: str | None = None
    frozen: bool = False
    softmin: float | None = None
    softmax: float | None = None
    hardmin: float | None = None
    hardmax: float | None = None
    delta: float | None = None


@dataclass
class XspecModel:
    """An Xspec model."""

    modeltype: ModelType
    name: str
    funcname: str
    language: LanguageStyle
    elo: float
    ehi: float
    parameters: Sequence[XspecParameter]
    use_errors: bool = False
    can_cache: bool = True


_info = {
    ##PYINFO##
}


def info(model: str) -> XspecModel:
    """Return information on the Xspec model from the model.dat file.

    This returns the information on the model as taken from the Xspec
    model library used to build this model.

    Parameters
    ----------
    name : str
        The Xspec model name (case-insensitive).

    Returns
    -------
    model : XspecModel
        The dataclass that describes the mode;.

    See Also
    --------
    list_models

    Examples
    --------

    >>> m = info('apec')
    >>> m.name
    'apec'
    >>> m.modeltype
    <ModelType.Add: 1>
    >>> [(p.name, p.default, p.units) for p in m.parameters]
    [('kT', 1.0, 'keV'), ('Abundanc', 1.0, None), ('Redshift', 0.0, None)]

    """
    # We want case-insensitive comparison but for the keys to retain
    # their case. Using casefold() rather than lower() is a bit OTT
    # here as I would bet model.dat is meant to be US-ASCII.
    model = str(model)
    check = model.casefold()
    out = next((v for k, v in _info.items() if k.casefold() == check), None)
    if out is None:
        raise ValueError(f"Unrecognized Xspec model '{model}'")

    return out


# Do we need Optional here?
def list_models(
    modeltype: ModelType | None = None, language: LanguageStyle | None = None
) -> list[str]:
    """Returns the names of Xspec models from the model.dat file.

    This returns the information on the model as taken from the Xspec
    model library used to build this model.

    Parameters
    ----------
    modeltype : ModelType or None, optional
        If not None then restrict the list to this model type.
    language : LanguageStyle or None, optional
        If not None then restrict the list to this language type.

    Returns
    -------
    models : list of str
        A sorted list of the model names.

    See Also
    --------
    info

    Notes
    -----
    The restrictions are combined, so setting both `modeltype` and
    `language` will select only those models which match both filters.

    Examples
    --------
    With Xspec 12.15.0:

    >>> len(list_models())
    310

    >>> 'tbabs' in list_models()
    True

    >>> 'TBabs' in list_models()
    False

    >>> list_models(modeltype=ModelType.Con)
    ['cflux', 'clumin', 'cpflux', 'gsmooth', ..., 'zashift', 'zmshift']

    >>> list_models(modeltype=ModelType.Con, language=LanguageStyle.F77Style4)
    ['kyconv', 'thcomp']

    """

    out = set()
    for k, v in _info.items():
        if modeltype is not None and v.modeltype != modeltype:
            continue

        if language is not None and v.language != language:
            continue

        out.add(k)

    return sorted(out)


if __INITIALIZED__:
    from .primitive import get_primitive as get_primitive
