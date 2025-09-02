from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import xspec
from numpy.testing import assert_allclose
from xspec import Xset

import xspex as xx
from xspex._compiled.lib.libxspex import clear_xflt_xspec, sync_xflt_to_xspec

from .conftest import XSPEC_ABUND_TABLES, XSPEC_XSECT_TABLES


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

    # Test spec_num can be either int or jnp.array
    fn_add(p, e, 1)
    fn_add(p, e, spec_num)

    # Test init_string check
    with pytest.raises(ValueError):
        fn_add(p, e, init_string=1)

    fn_con, _ = xx.get_model('cflux')
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


def test_abund_xsect_model_consistency():
    """Test abundance and cross-section settings affect model calculations.

    Comprehensive test that verifies abundance and cross-section table settings
    actually affect model calculations by comparing xspex phabs model output
    with PyXspec for all combinations of supported tables.

    This test ensures that:
        1. Different abundance tables produce different absorption spectra
        2. Different cross-section tables affect the calculations
        3. xspex maintains perfect consistency with XSPEC for all settings
        4. The settings are properly synchronized between xspex and PyXspec

    Uses the phabs (photoelectric absorption) model with various hydrogen
    column densities to test the effect of different atomic data tables.
    """
    # Get phabs model function
    phabs_fn, phabs_info = xx.get_model('phabs')

    # Set up test parameters
    # phabs parameter: nH (hydrogen column density)
    nh_values = [0.1, 1.0, 5.0]  # 10^22 cm^-2 units

    # Set up energy grid
    egrid = jnp.linspace(0.1, 10.0, 101, dtype=jnp.float64)
    egrid_list = egrid.tolist()

    # Save original settings
    original_abund = xx.abund()
    original_xsect = xx.xsect()

    # Test different abundance and cross-section combinations
    try:
        for abund in XSPEC_ABUND_TABLES:
            for xsect in XSPEC_XSECT_TABLES:
                # Set abundance and cross-section tables
                xx.abund(abund)
                xx.xsect(xsect)

                # Also set in PyXspec for comparison
                Xset.abund = abund
                Xset.xsect = xsect

                for nh in nh_values:
                    # Test parameters for phabs
                    params = jnp.array([nh], dtype=jnp.float64)
                    params_list = [nh]

                    # Get xspex result
                    val_xx = phabs_fn(params, egrid, 1)

                    # Get XSPEC result
                    val_xs = []
                    xspec.callModelFunction(
                        'phabs',
                        egrid_list,
                        params_list,
                        val_xs,
                    )

                    # Compare results
                    assert_allclose(
                        val_xx,
                        val_xs,
                        err_msg=(
                            f'phabs model mismatch with abund={abund}, '
                            f'xsect={xsect}, nH={nh}'
                        ),
                    )

    finally:
        # Restore original settings
        xx.abund(original_abund)
        xx.xsect(original_xsect)

        # Restore PyXspec settings
        original_abund_short = original_abund.split(':')[0]
        Xset.abund = original_abund_short
        Xset.xsect = original_xsect


def test_cosmo_model_consistency():
    """Test cosmological parameters affect model calculations, using clumin."""
    fn, _ = xx.get_model('clumin')
    p = jnp.array([1.0, 10.0, 1.0, 1.0], dtype=jnp.float64)
    e = jnp.linspace(0.1, 10.0, 101, dtype=jnp.float64)
    input_model = jnp.ones(e.size - 1)
    original_cosmo_xx = xx.cosmo()
    original_cosmo_xs = Xset.cosmo
    try:
        xx.cosmo(H0=67.4, q0=0.0, lambda0=0.68)
        Xset.cosmo = '67.4 0.0 0.68'
        xx_val = fn(p, e, input_model, 1)
        xs_val = input_model.tolist()
        xspec.callModelFunction('clumin', e.tolist(), p.tolist(), xs_val)
        assert_allclose(xx_val, xs_val)
    finally:
        # Restore original cosmological parameters
        xx.cosmo(**original_cosmo_xx)
        Xset.cosmo = ' '.join(original_cosmo_xs)

    # Test cosmological parameters are correctly reset to original values
    xx_val = fn(p, e, input_model, 1)
    xs_val = input_model.tolist()
    xspec.callModelFunction('clumin', e.tolist(), p.tolist(), xs_val)
    assert_allclose(xx_val, xs_val)


def test_mstr_model_consistency():
    """Test model strings affect model calculations, using powerlaw."""
    fn, _ = xx.get_model('powerlaw')
    p = jnp.array([1.0], dtype=jnp.float64)
    e = jnp.linspace(0.1, 10.0, 101, dtype=jnp.float64)
    test_mstr = {'POW_EMIN': '1.0', 'POW_EMAX': '10.0'}
    try:
        xx.mstr(test_mstr)
        Xset.modelStrings = test_mstr
        xx_val = fn(p, e, 1)
        xs_val = []
        xspec.callModelFunction('powerlaw', e.tolist(), p.tolist(), xs_val)
        assert_allclose(xx_val, xs_val)
    finally:
        # Clear all model strings
        xx.clear_mstr()
        Xset.modelStrings = {}


@pytest.mark.parametrize(
    'model, params',
    [
        pytest.param('polconst', [1.0, 30.0], id='polconst'),
        pytest.param('pollin', [1.0, 1.0, 30.0, 1.0], id='pollin'),
        pytest.param('polpow', [1.0, 1.0, 30.0, 1.0], id='polpow'),
    ],
)
def test_xflt_model_consistency(model, params):
    """Test XFLT affect model calculations, using polarization models."""
    xflt = {1: {'Stokes': 0.0}, 2: {'Stokes': 1.0}, 3: {'Stokes': 2.0}}
    xx.xflt(xflt)
    sync_xflt_to_xspec()

    fn, _ = xx.get_model(model)
    p = jnp.array(params, dtype=jnp.float64)
    e = jnp.linspace(0.1, 10.0, 101, dtype=jnp.float64)
    xx_val = fn(p, e, 1)
    xs_val = []
    xspec.callModelFunction(model, e.tolist(), p.tolist(), xs_val)
    assert_allclose(xx_val, xs_val)

    xx.clear_xflt()
    clear_xflt_xspec()
