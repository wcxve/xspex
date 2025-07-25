from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from jax import numpy as jnp
from mxspec import callModelFunction
from numpy.testing import assert_allclose
from xspec import Xset

import xspex as xx
from xspex._xspec.types import XspecModelType

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from xspex._xspec.types import XspecParam


def test_xspec_version():
    assert xx.xspec_version() == Xset.version[1]


def test_abund():
    assert xx.abund() == Xset.abund[:4]


def test_xsect():
    assert xx.xsect() == Xset.xsect


def test_cosmo():
    keys = ['H0', 'q0', 'lambda0']
    xx_cosmo = xx.cosmo()
    xs_cosmo = Xset.cosmo
    for i, key in enumerate(keys):
        assert_allclose(
            np.float32(xx_cosmo[key]),
            np.float32(xs_cosmo[i]),
            err_msg=key,
        )


def get_default_pars(param: XspecParam) -> float:
    if param.name.casefold() == 'redshift':
        return 1.0
    elif param.name.casefold() == 'velocity':
        return 100.0
    else:
        return float(param.default)


def get_model_eval_args(
    name: str,
) -> tuple[
    Callable,
    str,
    Array,
    Array,
    tuple[float, ...],
    tuple[float, ...],
    bool,
    bool,
]:
    fn, info = xx.get_model(name)
    egrid = jnp.linspace(0.1, 10, 100)
    egrid_tuple = tuple(egrid.tolist())
    pars_tuple = tuple(get_default_pars(p) for p in info.parameters)
    pars = jnp.array(pars_tuple)
    return (
        fn,
        info.name,
        pars,
        egrid,
        pars_tuple,
        egrid_tuple,
        info.data_depend,
        info.type == XspecModelType.Con,
    )


@pytest.mark.parametrize('name', xx.list_models())
def test_model_eval(name: str):
    """Test model evaluation against PyXspec."""
    fn, mname, p, e, p_, e_, data_depend, is_con = get_model_eval_args(name)

    if data_depend:
        pytest.skip('data-dependent model, requires XFLT setup')

    if not is_con:
        val_xx = fn(p, e, 1)
        callModelFunction(mname, e_, p_, val_xs := [])
    else:
        n_model = len(e) - 1
        val_xx = fn(p, e, jnp.ones(n_model), 1)
        val_xs = [1.0] * n_model
        callModelFunction(mname, e_, p_, val_xs)
    assert_allclose(val_xx, val_xs)
