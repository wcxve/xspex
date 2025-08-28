"""XSPEC model definitions.

References
----------
.. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSappendixLocal.html
.. [2] https://github.com/cxcsds/parse_xspec
.. [3] https://github.com/cxcsds/xspec-models-cxc
.. [4] https://github.com/cxcsds/xspec-models-cxc-helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class XspecFuncType(Enum):
    """XSPEC model function types, see [1]_ for details.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSappendixLocal.html
    """

    CXX = auto()
    """C++ style function in double precision."""

    C = auto()
    """C style function in double precision."""

    f = auto()
    """Fortran function in single precision."""

    F = auto()
    """Fortran function in double precision."""

    @classmethod
    def get_func_and_type(cls, name: str) -> tuple[str, XspecFuncType]:
        """Get function and type given the name in model.dat file."""
        match name[:2]:
            case 'C_':
                return name[2:], cls.CXX
            case 'c_':
                return name[2:], cls.C
            case 'F_':
                return f'{name[2:]}_', cls.F
            case _:
                return f'{name}_', cls.f


class XspecModelType(Enum):
    """XSPEC model types."""

    Add = auto()
    """XSPEC additive models [1]_.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/Additive.html
    """

    Mul = auto()
    """XSPEC multiplicative models [1]_.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/Multiplicative.html
    """

    Con = auto()
    """XSPEC convolution models [1]_.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/Convolution.html
    """

    Mix = auto()
    """XSPEC mixing models [1]_. Currently not supported.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/Mixing.html
    """

    Acn = auto()
    """XSPEC pile-up models [1]_. Currently not supported.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/Pileup.html
    """

    Amx = auto()
    """A combination of mixing and pile-up models. Currently not supported."""

    @classmethod
    def from_str(cls, s: str) -> XspecModelType:
        """Get model type from model name."""
        match s.casefold():
            case 'add':
                return cls.Add
            case 'mul':
                return cls.Mul
            case 'con':
                return cls.Con
            case 'mix':
                return cls.Mix
            case 'acn':
                return cls.Acn
            case 'amx':
                return cls.Amx
            case _:
                raise ValueError(f'invalid model type: {s}')


class XspecParamType(Enum):
    """XSPEC parameter types, see [1]_ for details.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSappendixLocal.html
    """

    Basic = auto()
    """Basic parameter."""

    Switch = auto()
    """Switch parameter, requires integer value."""

    Scale = auto()
    """Scale parameter."""


@dataclass(frozen=True)
class XspecParam:
    """XSPEC parameter, see [1]_ and [2]_ for details.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSnewpar.html
    .. [2] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSappendixLocal.html
    """

    type: XspecParamType
    """Parameter type."""

    name: str
    """Parameter name."""

    unit: str
    """Parameter unit."""

    default: float
    """Parameter default value."""

    min: float | None
    """Parameter hard minimum."""

    bot: float | None
    """Parameter soft minimum."""

    top: float | None
    """Parameter soft maximum."""

    max: float | None
    """Parameter hard maximum."""

    delta: float | None
    """Finite difference step size in computing numerical derivatives."""

    fixed: bool
    """Whether the parameter is fixed."""

    periodic: bool
    """Whether the parameter is periodic."""


@dataclass(frozen=True)
class XspecModel:
    """XSPEC model, see [1]_ for details.

    References
    ----------
    .. [1] https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/xanadu/xspec/manual/XSappendixLocal.html
    """

    type: XspecModelType
    """Model type."""

    name: str
    """Model name."""

    func: str
    """Model function name."""

    func_type: XspecFuncType
    """Model function type."""

    desc: str
    """Model description."""

    link: str
    """Online XSPEC documentation link for the model."""

    emin: float
    """The minimum valid energy of the model."""

    emax: float
    """The maximum valid energy of the model."""

    parameters: tuple[XspecParam, ...]
    """Model parameters."""

    calc_errors: bool
    """Whether the model calculates errors."""

    data_depend: bool
    """Whether the model is data dependent."""

    init_string: str
    """Initialization string of the model."""

    @property
    def n_params(self) -> int:
        """Number of parameters of the model."""
        return len(self.parameters)
