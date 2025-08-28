from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xspec
from jax.sharding import NamedSharding, PartitionSpec as P
from numpy.testing import assert_allclose

import xspex as xx
from xspex._xspec.types import XspecModelType

from .conftest import MODELS_XFAIL

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array
    from numpy.typing import NDArray

    from xspex._xspec.types import XspecParam


N_DEVICES = len(jax.devices())
N_BATCHES = 2
SHARDING = NamedSharding(
    jax.make_mesh((N_DEVICES, 1), ('devices', 'batch')),
    P('devices', 'batch'),
)
REL_DELTA = 0.2


def get_pars(
    param: XspecParam,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Get randomized parameter values for testing."""
    shape = (N_DEVICES, N_BATCHES)
    if param.fixed:
        return np.full(shape, float(param.default))
    elif param.name.casefold() == 'redshift':
        return rng.uniform(0.1, 5, shape)
    elif param.name.casefold() == 'velocity':
        return rng.uniform(-1e3, 1e3, shape)
    else:
        if param.min is None or param.max is None:
            return np.full(shape, float(param.default))
        p_default = param.default
        p_abs = abs(p_default)
        p_min = param.min
        p_max = param.max
        return rng.uniform(
            max(p_default - REL_DELTA * p_abs, p_min),
            min(p_default + REL_DELTA * p_abs, p_max),
            shape,
        )


def get_model_eval_args(
    name: str,
    seed: int,
) -> tuple[
    Callable,
    str,
    Array,
    Array,
    list[float],
    bool,
    bool,
]:
    """Get model evaluation arguments for testing."""
    fn, info = xx.get_model(name)
    # clamp to a reasonable testing range
    emin = max(0.01, info.emin)
    emax = min(100.0, info.emax)
    if not (emax > emin):
        pytest.skip(
            f'invalid energy range for {name}: [{info.emin}, {info.emax}]'
        )
    egrid = jnp.linspace(emin, emax, 101)
    egrid_ = egrid.tolist()
    rng = np.random.default_rng(seed)
    pars = np.array([get_pars(p, rng) for p in info.parameters])
    pars = np.moveaxis(pars, 0, -1)

    # Handle models with no parameters
    if len(pars) == 0:
        pars = np.tile(np.empty((0,)), (N_DEVICES, N_BATCHES, 0))

    # some models have paramters need to be constrained
    if name == 'xion':
        # outer radius > inner radius > height of source
        pars[..., 4] += pars[..., 0]
        pars[..., 5] += pars[..., 4]

    return (
        fn,
        info.name,
        jnp.array(pars),
        egrid,
        egrid_,
        info.data_depend,
        info.type == XspecModelType.Con,
    )


@pytest.mark.parametrize(
    'name, seed',
    [pytest.param(m, i, id=m) for i, m in enumerate(xx.list_models())],
)
def test_vmap(name: str, seed: int):
    """Test jax.vmap consistency with XSPEC for all models."""
    fn, mname, p, e, e_, data_depend, is_con = get_model_eval_args(name, seed)

    if data_depend:
        pytest.skip('data-dependent model, requires XFLT setup')

    params_batch = jax.device_put(p, SHARDING)

    try:
        if not is_con:
            # Test additive/multiplicative models
            # Vectorize over parameter dimension
            vmap_fn = jax.vmap(fn, in_axes=(0, None, None))
            vmap2_fn = jax.vmap(vmap_fn, in_axes=(0, None, None))
            val_xx_batch = jax.jit(vmap2_fn)(params_batch, e, 1)

            # Compare with individual XSPEC calls
            for i in range(N_DEVICES):
                for j in range(N_BATCHES):
                    val_xs = []
                    xspec.callModelFunction(
                        mname,
                        e_,
                        params_batch[i, j].tolist(),
                        val_xs,
                    )
                    assert_allclose(
                        val_xx_batch[i, j],
                        val_xs,
                        err_msg=(
                            f'pars={p[i, j].tolist()}, '
                            f'emin={e[0]}, emax={e[-1]}'
                        ),
                    )
        else:
            # Test convolution models
            n_model = len(e) - 1
            input_model = jnp.ones(n_model)

            # Vectorize over parameter dimension
            vmap_fn = jax.vmap(fn, in_axes=(0, None, None, None))
            vmap2_fn = jax.vmap(vmap_fn, in_axes=(0, None, None, None))
            val_xx_batch = jax.jit(vmap2_fn)(params_batch, e, input_model, 1)

            # Compare with individual XSPEC calls
            for i in range(N_DEVICES):
                for j in range(N_BATCHES):
                    val_xs = [1.0] * n_model
                    xspec.callModelFunction(
                        mname,
                        e_,
                        params_batch[i, j].tolist(),
                        val_xs,
                    )
                    assert_allclose(
                        val_xx_batch[i, j],
                        val_xs,
                        err_msg=(
                            f'pars={p[i, j].tolist()}, '
                            f'emin={e[0]}, emax={e[-1]}'
                        ),
                    )

    except AssertionError as e:
        if name in MODELS_XFAIL:
            pytest.xfail('occasional failures')
        raise e


@pytest.mark.parametrize(
    'name, seed',
    [pytest.param(m, i, id=m) for i, m in enumerate(xx.list_models())],
)
def test_pmap(name: str, seed: int):
    """Test jax.pmap consistency with XSPEC for all models."""
    fn, mname, p, e, e_, data_depend, is_con = get_model_eval_args(name, seed)

    if data_depend:
        pytest.skip('data-dependent model, requires XFLT setup')

    try:
        if not is_con:
            # Test additive/multiplicative models
            # Parallelize over parameter sets
            vmap_fn = jax.vmap(fn, in_axes=(0, None, None))
            pmap_fn = jax.pmap(vmap_fn, in_axes=(0, None, None))
            val_xx = pmap_fn(p, e, 1)

            # Compare with individual XSPEC calls
            for i in range(N_DEVICES):
                for j in range(N_BATCHES):
                    val_xs = []
                    xspec.callModelFunction(
                        mname,
                        e_,
                        p[i, j].tolist(),
                        val_xs,
                    )
                    assert_allclose(
                        val_xx[i, j],
                        val_xs,
                        err_msg=(
                            f'pars={p[i, j].tolist()}, '
                            f'emin={e[0]}, emax={e[-1]}'
                        ),
                    )
        else:
            # Test convolution models
            n_model = len(e) - 1
            input_model = jnp.ones(n_model)

            # Parallelize over parameter sets
            vmap_fn = jax.vmap(fn, in_axes=(0, None, None, None))
            pmap_fn = jax.pmap(vmap_fn, in_axes=(0, None, None, None))
            val_xx = pmap_fn(p, e, input_model, 1)

            # Compare with individual XSPEC calls
            for i in range(N_DEVICES):
                for j in range(N_BATCHES):
                    val_xs = [1.0] * n_model
                    xspec.callModelFunction(
                        mname,
                        e_,
                        p[i, j].tolist(),
                        val_xs,
                    )
                    assert_allclose(
                        val_xx[i, j],
                        val_xs,
                        err_msg=(
                            f'pars={p[i, j].tolist()}, '
                            f'emin={e[0]}, emax={e[-1]}'
                        ),
                    )

    except AssertionError as e:
        if name in MODELS_XFAIL:
            pytest.xfail('occasional failures')
        raise e
