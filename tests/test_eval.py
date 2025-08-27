from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xspec
from numpy.testing import assert_allclose

import xspex as xx


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
