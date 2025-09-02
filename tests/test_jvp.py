import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

import xspex as xx


def get_model_eval_args(model_name: str):
    fn, model_info = xx.get_model(model_name)
    params = jnp.array([p.default for p in model_info.parameters])
    emin = max(0.01, model_info.emin)
    emax = min(100.0, model_info.emax)
    egrid = jnp.linspace(emin, emax, 101)
    return fn, params, egrid


def test_fdjvp():
    """Test JVP for some models."""
    fn, p, e = get_model_eval_args('apec')
    jacfwd = jax.jacfwd(fn)(p, e)
    jacrev = jax.jacrev(fn)(p, e)
    assert_allclose(jacfwd, jacrev)

    fn, p, e = get_model_eval_args('cflux')
    input_model = jnp.ones(len(e) - 1)
    jacfwd = jax.jacfwd(fn)(p, e, input_model)
    jacrev = jax.jacrev(fn)(p, e, input_model)
    assert_allclose(jacfwd, jacrev)
