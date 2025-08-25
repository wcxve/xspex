from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from jax import numpy as jnp
from mxspec import callModelFunction
from numpy.testing import assert_allclose

import xspex as xx
from xspex._xspec.types import XspecModelType

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from xspex._xspec.types import XspecParam


def test_check_input():
    fn_add, info = xx.get_model('powerlaw')
    p = jnp.array([1.0])
    e = jnp.array([1.0, 2.0])
    spec_num = jnp.array(1)

    # Test params dimension check
    with pytest.raises(ValueError):
        fn_add(p[0], e, spec_num)
    with pytest.raises(ValueError):
        fn_add(jnp.atleast_2d(p), e, spec_num)

    # Test params length check
    with pytest.raises(ValueError):
        fn_add(jnp.append(p, p), e, spec_num)

    # Test params dtype check
    with pytest.raises(ValueError):
        fn_add(p.astype(jnp.float32), e, spec_num)
    with pytest.raises(ValueError):
        fn_add(p.astype(int), e, spec_num)

    # Test egrid dimension check
    with pytest.raises(ValueError):
        fn_add(p, e[0], spec_num)
    with pytest.raises(ValueError):
        fn_add(p, jnp.atleast_2d(e), spec_num)

    # Test egrid length check
    with pytest.raises(ValueError):
        fn_add(p, jnp.array([1.0]), spec_num)

    # Test egrid dtype check
    with pytest.raises(ValueError):
        fn_add(p, e.astype(jnp.float32), spec_num)
    with pytest.raises(ValueError):
        fn_add(p, e.astype(int), spec_num)

    # Test spec_num scalar check
    with pytest.raises(ValueError):
        fn_add(p, e, jnp.atleast_1d(spec_num))
    with pytest.raises(ValueError):
        fn_add(p, e, jnp.atleast_2d(spec_num))
    with pytest.raises(ValueError):
        fn_add(p, e, jnp.append(spec_num, spec_num))

    # Test spec_num dtype check
    with pytest.raises(ValueError):
        fn_add(p, e, spec_num.astype(jnp.int32))
    with pytest.raises(ValueError):
        fn_add(p, e, spec_num.astype(float))

    fn_con, info = xx.get_model('cflux')
    p = jnp.array([1.0, 2.0, 1.0])
    e = jnp.array([1.0, 2.0])
    m = jnp.array([1.0])
    spec_num = jnp.array(1)

    # Test input model dimension check
    with pytest.raises(ValueError):
        fn_con(p, e, m[0], spec_num)
    with pytest.raises(ValueError):
        fn_con(p, e, jnp.atleast_2d(m), spec_num)

    # Test input model dtype check
    with pytest.raises(ValueError):
        fn_con(p, e, m.astype(jnp.float32), spec_num)
    with pytest.raises(ValueError):
        fn_con(p, e, m.astype(int), spec_num)


# According to xspec-models-cxc:
# grbjet occasional failures has been reported to XSPEC (in 12.12.0)
MODELS_SKIP = ['grbjet']


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

    if name in MODELS_SKIP:
        pytest.skip(f'model {name} is skipped')

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
