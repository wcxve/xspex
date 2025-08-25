from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
import xspec
from numpy.testing import assert_allclose

import xspex as xx
from xspex._xspec.types import XspecModelType

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from xspex._xspec.types import XspecParam


def test_check_input():
    fn_add, info = xx.get_model('powerlaw')
    p = jnp.array([1.0], dtype=jnp.float64)
    e = jnp.array([1.0, 2.0], dtype=jnp.float64)
    spec_num = jnp.array(1, dtype=jnp.int64)

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
    p = jnp.array([1.0, 2.0, 1.0], dtype=jnp.float64)
    e = jnp.array([1.0, 2.0], dtype=jnp.float64)
    m = jnp.array([1.0], dtype=jnp.float64)
    spec_num = jnp.array(1, dtype=jnp.int64)

    # Test input model dimension check
    with pytest.raises(ValueError):
        fn_con(p, e, m[0], spec_num)
    with pytest.raises(ValueError):
        fn_con(p, e, jnp.atleast_2d(m), spec_num)

    # Test input model length check
    with pytest.raises(ValueError):
        fn_con(p, e, jnp.append(m, m), spec_num)

    # Test input model dtype check
    with pytest.raises(ValueError):
        fn_con(p, e, m.astype(jnp.float32), spec_num)
    with pytest.raises(ValueError):
        fn_con(p, e, m.astype(int), spec_num)


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
    list[float],
    list[float],
    bool,
    bool,
]:
    fn, info = xx.get_model(name)
    # clamp to a reasonable testing range
    emin = max(0.01, info.emin)
    emax = min(100.0, info.emax)
    if not (emax > emin):
        pytest.skip(
            f'invalid energy range for {name}: [{info.emin}, {info.emax}]'
        )
    egrid = jnp.linspace(emin, emax, 201)
    egrid_ = egrid.tolist()
    pars_ = [get_default_pars(p) for p in info.parameters]
    pars = jnp.array(pars_)
    return (
        fn,
        info.name,
        pars,
        egrid,
        pars_,
        egrid_,
        info.data_depend,
        info.type == XspecModelType.Con,
    )


# models with occasional failures
MODELS_XFAIL = (
    'grbjet',
    'ismdust',
)


@pytest.mark.parametrize('name', xx.list_models())
def test_model_eval(name: str):
    """Test model evaluation against PyXspec."""
    fn, mname, p, e, p_, e_, data_depend, is_con = get_model_eval_args(name)

    if data_depend:
        pytest.skip('data-dependent model, requires XFLT setup')

    if not is_con:
        val_xx = fn(p, e, 1)
        xspec.callModelFunction(mname, e_, p_, val_xs := [])
    else:
        n_model = len(e) - 1
        val_xx = fn(p, e, jnp.ones(n_model), 1)
        val_xs = [1.0] * n_model
        xspec.callModelFunction(mname, e_, p_, val_xs)
    try:
        assert_allclose(
            val_xx,
            val_xs,
            err_msg=f'diff at {np.flatnonzero(~np.isclose(val_xx, val_xs))}',
        )
    except AssertionError as e:
        if name in MODELS_XFAIL:
            pytest.xfail('occasional failures')
        raise e


def test_larger_egrid_size_eval():
    """Test model evaluation with a larger egrid size.

    This test is to check if the model evaluation is correct when the egrid
    size is larger than the default size, which will trigger the memory
    allocation in the xspex worker.
    """
    pl, _ = xx.get_model('powerlaw')

    egrid = jnp.linspace(0.01, 100.0, 8192)
    p = jnp.array([1.0], dtype=jnp.float64)
    val_xx = pl(p, egrid, 1)
    xspec.callModelFunction(
        'powerlaw', egrid.tolist(), p.tolist(), val_xs := []
    )
    assert_allclose(
        val_xx,
        val_xs,
        err_msg=f'diff at {np.flatnonzero(~np.isclose(val_xx, val_xs))}',
    )

    cflux, _ = xx.get_model('cflux')
    p = jnp.array([1.0, 100.0, 1.0], dtype=jnp.float64)
    val_xx = cflux(p, egrid, val_xx, 1)
    xspec.callModelFunction('cflux', egrid.tolist(), p.tolist(), val_xs)
    assert_allclose(
        val_xx,
        val_xs,
        err_msg=f'diff at {np.flatnonzero(~np.isclose(val_xx, val_xs))}',
    )
