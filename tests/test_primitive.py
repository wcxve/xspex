import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xspex as x

jax.config.update('jax_enable_x64', True)

# We need to set up the cosmology as this is currently not done in FNINIT.
#
x.cosmology(H0=70, lambda0=0.73, q0=0)

# We want to ensure we have a fixed abundance / cross section
# for the checks below. If we didn't set them here then it
# would depend on the user's ~/.xspec/Xspec.init file
#
x.abundance('lodd')
x.cross_section('vern')
version = x.version().split('.')
major = int(version[0])
minor = int(version[1])
if major > 12 or (major == 12 and minor >= 15):
    x.set_model_string('NEIVERS', '3.1.2')


def test_powerlaw_primitive():
    @jax.jit
    def pl(p, e, _):
        p = 1 - p
        f = e**p / p
        return f[1:] - f[:-1]

    powerlaw, _ = x.get_primitive('powerlaw')
    params = jnp.array([1.5])
    egrid = jnp.arange(1.0, 10.0, 0.1)

    res_true = pl(params, egrid, 1)
    res_test1 = powerlaw(params, egrid, 1)
    res_test2 = jax.jit(powerlaw)(params, egrid, 1)

    assert jnp.allclose(res_true, res_test1)
    assert jnp.allclose(res_true, res_test2)

    prim_in = (jnp.r_[1.1], jnp.linspace(1.1, 1.5, 6), 1)
    tan_in = (
        jnp.r_[1.0],
        jnp.zeros_like(prim_in[1]),
        np.zeros((), dtype=jax.dtypes.float0),
    )
    jvp_true = jax.jvp(pl, prim_in, tan_in)
    jvp_test = jax.jvp(powerlaw, prim_in, tan_in)
    assert jnp.allclose(jvp_true[0], jvp_test[0])
    assert jnp.allclose(jvp_true[1], jvp_test[1])

    args = (
        jnp.expand_dims(jnp.linspace(-3, 3, 101), axis=-1),
        jnp.geomspace(0.1, 20, 10001),
        1,
    )
    f1 = jax.vmap(jax.jit(powerlaw), (0, None, None), 0)
    f2 = jax.vmap(pl, (0, None, None), 0)
    assert jnp.allclose(f1(*args), f2(*args))


# Unfortunately some models need to be skipped for
# some reason. This is obviously going to be version-specific.
#
# grbjet occasional failures has been reported to XSPEC (in 12.12.0).
#
# rfxconv and xilconv require additinal setup (e.g. energy range) that I have
# no energy to diesntangle, so we skip
#
# rgsxsrc requires extra setup (an image file)
#
MODELS_ADD = x.list_models(modeltype=x.ModelType.Add)
MODELS_MUL = x.list_models(modeltype=x.ModelType.Mul)
MODELS_CON = x.list_models(modeltype=x.ModelType.Con)

MODELS_ADD_SKIP = ['grbjet']
MODELS_MUL_SKIP = []
MODELS_CON_SKIP = ['rfxconv', 'rgsext', 'rgsxsrc', 'xilconv']


@pytest.mark.parametrize('model', MODELS_ADD + MODELS_MUL)
def test_eval(model):
    p, info = x.get_primitive(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_ADD_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    mfunc = getattr(x, model)

    pars = [
        0.1 if p.name.casefold() == 'redshift' else p.default
        for p in info.parameters
    ]
    pars = np.array(pars)

    egrid = np.arange(0.1, 10, 0.01)
    y1 = mfunc(energies=egrid, pars=pars)
    y2 = p(pars, egrid, 1)
    y3 = jax.jit(p)(pars, egrid, 1)

    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)


@pytest.mark.parametrize('model', MODELS_CON)
def test_eval_con(model):
    """Evaluate a convolution additive model."""
    p, info = x.get_primitive(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_CON_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    egrid = np.arange(0.1, 10, 0.01)

    def conv(p):
        """what is the default parameter value?"""

        if p.name.casefold() == 'redshift':
            return 0.01

        if p.name.casefold() == 'velocity':
            return 100.0

        return p.default

    pars = [p.default for p in x.info('powerlaw').parameters]
    pars = np.array(pars)
    mvals = x.powerlaw(energies=egrid, pars=pars)
    # mvals = mvals.astype(get_dtype(info))
    ymodel = mvals.copy()

    mfunc = getattr(x, model)

    pars = np.array([conv(p) for p in info.parameters])
    y1 = mfunc(energies=egrid, pars=pars, model=mvals)
    y2 = p(pars, egrid, mvals, 1)
    y3 = jax.jit(p)(pars, egrid, mvals, 1)

    assert (y1 > 0).any()
    assert np.any(y1 != ymodel)
    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)


@pytest.mark.parametrize('model', MODELS_ADD + MODELS_MUL)
def test_batching(model):
    p, info = x.get_primitive(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_ADD_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    if model in ['feklor', 'posm']:
        pytest.skip(f'Model {model} has no shape parameter.')

    mfunc = getattr(x, model)

    n = 5
    pars = [
        np.full(n, 0.1)
        if p.name.casefold() == 'redshift'
        else np.full(n, p.default)
        for p in info.parameters
    ]
    pars = np.column_stack(pars)

    egrid = np.arange(0.1, 10, 0.01)
    y1 = np.array([mfunc(energies=egrid, pars=p) for p in pars])
    y2 = jax.vmap(p, in_axes=(0, None, None), out_axes=0)(pars, egrid, 1)
    y3 = jax.jit(jax.vmap(p, in_axes=(1, None, None), out_axes=0))(
        pars.T, egrid, 1
    )

    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)


@pytest.mark.parametrize('model', MODELS_CON)
def test_batching_con(model):
    """Evaluate a convolution additive model."""
    p, info = x.get_primitive(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_CON_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    egrid = np.arange(0.1, 10, 0.01)

    def conv(p):
        """what is the default parameter value?"""

        if p.name.casefold() == 'redshift':
            return 0.01

        if p.name.casefold() == 'velocity':
            return 100.0

        return p.default

    pars = [p.default for p in x.info('powerlaw').parameters]
    pars = np.array(pars)
    mvals = x.powerlaw(energies=egrid, pars=pars)
    # mvals = mvals.astype(get_dtype(info))
    ymodel = mvals.copy()

    mfunc = getattr(x, model)

    n = 5
    pars = np.column_stack([np.full(n, conv(p)) for p in info.parameters])
    y1 = np.array([mfunc(energies=egrid, pars=p, model=mvals) for p in pars])
    y2 = jax.vmap(p, in_axes=(0, None, None, None), out_axes=0)(
        pars, egrid, mvals, 1
    )
    y3 = jax.jit(jax.vmap(p, in_axes=(0, None, None, None), out_axes=0))(
        pars, egrid, mvals, 1
    )
    mvals_n = np.tile(mvals, (n, 1))
    y4 = jax.vmap(p, in_axes=(None, None, 0, None), out_axes=0)(
        pars[0], egrid, mvals_n, 1
    )
    y5 = jax.vmap(p, in_axes=(0, None, 0, None), out_axes=0)(
        pars, egrid, mvals_n, 1
    )

    assert (y1 > 0).any()
    assert np.any(y1 != ymodel)
    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)
    assert np.allclose(y1, y4)
    assert np.allclose(y1, y5)


# @pytest.mark.parametrize("model", MODELS_ADD + MODELS_MUL)
# def test_jvp(model):
#     p, info = x.get_primitive(model)
#     if not info.can_cache:
#         pytest.skip(f"Model {model} can not be cached.")
#
#     if model in MODELS_ADD_SKIP:
#         pytest.skip(f"Model {model} is marked as un-testable.")
#
#     if model == 'posm':
#         pytest.skip(f"Model {model} has no shape parameter.")
#
#     if model in ['nsa', 'nsmax', 'nsmaxg', 'nsx']:
#         pytest.skip()
#
#     pars = np.array([0.1 if p.name.casefold() == 'redshift' else p.default
#                      for p in info.parameters])
#
#     egrid = np.arange(0.1, 10, 0.01)
#
#     prim_in = (pars, egrid, 1)
#     tan_in = (
#         np.ones_like(pars),
#         np.ones_like(egrid),
#         np.zeros((), dtype=jax.dtypes.float0)
#     )
#
#     tan = jax.jvp(p, prim_in, tan_in)[1]
#
#     assert jnp.isfinite(tan).all()
#     assert jnp.isnan(tan).sum() == 0
