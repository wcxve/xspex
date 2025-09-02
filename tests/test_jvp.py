import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

import xspex as xx


def test_fd_method():
    fn, model_info = xx.get_model('apec')

    # Test unsupported method
    with pytest.raises(ValueError):
        xx.define_fdjvp(fn, model_info, method='backward')

    p = jnp.array([1.0, 1.0, 1.0])
    e = jnp.linspace(0.1, 1.0, 10)

    # Test forward method
    xx.define_fdjvp(fn, model_info, method='forward')
    jacfwd_f = jax.jacfwd(fn)(p, e)
    jacrev_f = jax.jacrev(fn)(p, e)
    assert_allclose(jacfwd_f, jacrev_f)

    # Test central method
    xx.define_fdjvp(fn, model_info, method='central')
    jacfwd_c = jax.jacfwd(fn)(p, e)
    jacrev_c = jax.jacrev(fn)(p, e)
    assert_allclose(jacfwd_c, jacrev_c)

    # Test forward and central methods are consistent
    assert_allclose(jacfwd_f, jacfwd_c)


def test_fd_fixed():
    # Test fixed size check
    for model_name in ['apec', 'posm']:
        fn, model_info = xx.get_model(model_name)
        n_params = model_info.n_params
        with pytest.raises(ValueError, match='size'):
            xx.define_fdjvp(fn, model_info, fixed=[False] * (n_params + 1))

    # Test scale and switch parameters must be fixed
    for model_name in ['cie', 'logpar']:
        fn, model_info = xx.get_model(model_name)
        n_params = model_info.n_params
        with pytest.raises(ValueError, match='must be fixed'):
            xx.define_fdjvp(fn, model_info, fixed=[False] * n_params)

    # Test when all parameters are fixed, the JVP should be zero
    fn, model_info = xx.get_model('apec')
    n_params = model_info.n_params
    fn = xx.define_fdjvp(fn, model_info, fixed=[True] * n_params)
    jacfwd = jax.jacfwd(fn)(jnp.ones(n_params), jnp.linspace(0.1, 1.0, 10))
    assert_allclose(jacfwd, jnp.zeros_like(jacfwd))


def test_fd_delta():
    # Test delta must be finite
    with pytest.raises(ValueError, match='finite'):
        xx.define_fdjvp(*xx.get_model('apec'), delta=jnp.inf)
    with pytest.raises(ValueError, match='NaN'):
        xx.define_fdjvp(*xx.get_model('apec'), delta=jnp.nan)

    # Small delta should be consistent with delta=0
    fn, model_info = xx.get_model('apec')
    p = jnp.array([1.0, 1.0, 1.0])
    e = jnp.linspace(0.1, 1.0, 10)
    jac = jax.jacfwd(fn)(p, e)

    fn = xx.define_fdjvp(fn, model_info, delta=0.01)
    assert_allclose(jac, jax.jacfwd(fn)(p, e))

    # Test using step size from model info
    fn = xx.define_fdjvp(fn, model_info, delta=-1)
    assert_allclose(jac, jax.jacfwd(fn)(p, e))

    # Test fd JVP is correct for zero-value parameters
    def pl(p, e):
        a = p[0]
        one_minus_a = 1.0 - a
        return jnp.diff(e**one_minus_a / one_minus_a)

    fn, model_info = xx.get_model('powerlaw')
    p = jnp.zeros(1)
    e = jnp.linspace(0.1, 1.0, 10)
    assert_allclose(jax.jacfwd(fn)(p, e), jax.jacfwd(pl)(p, e))
