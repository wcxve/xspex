import jax
import jax.numpy as jnp
import numpy as np

from xspex import get_primitive

jax.config.update('jax_enable_x64', True)


def test_primitive():
    @jax.jit
    def pl(p, e, _):
        p = 1 - p
        f = e ** p / p
        return f[1:] - f[:-1]

    powerlaw, _ = get_primitive('powerlaw')
    params = jnp.array([1.5])
    egrid = jnp.arange(1., 10., 0.1)

    res_true = pl(params, egrid, 1)
    res_test1 = powerlaw(params, egrid, 1)
    res_test2 = jax.jit(powerlaw)(params, egrid, 1)

    assert jnp.allclose(res_true, res_test1)
    assert jnp.allclose(res_true, res_test2)

    prim_in = (jnp.r_[1.1], jnp.linspace(1.1, 1.5, 6), 1)
    tan_in = (
        jnp.r_[1.0],
        jnp.zeros_like(prim_in[1]),
        np.zeros((), dtype=jax.dtypes.float0)
    )
    jvp_true = jax.jvp(pl, prim_in, tan_in)
    jvp_test = jax.jit(lambda p, t: jax.jvp(powerlaw, prim_in, tan_in))(prim_in, tan_in)
    assert jnp.allclose(jvp_true[0], jvp_test[0])
    assert jnp.allclose(jvp_true[1], jvp_test[1])

    args = (
        jnp.ones((2, 1)) * 1.1,
        jnp.geomspace(0.1, 20, 10001),
        1,
    )
    f1 = jax.vmap(powerlaw, (0, None, None), 0)
    f2 = jax.vmap(pl, (0, None, None), 0)
    assert jnp.allclose(f1(*args), f2(*args))
