from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.flatten_util import ravel_pytree

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


def define_fdjvp(
    fn: Callable,
    method: Literal['central', 'forward'] = 'central',
) -> Callable:
    """Define JVP using finite differences."""
    if method not in {'central', 'forward'}:
        raise ValueError(
            f"supported methods are 'central' and 'forward', but got "
            f"'{method}'"
        )

    def fdjvp(primals, tangents):
        egrid, params = primals
        egrid_tangent, params_tangent = tangents

        if not isinstance(egrid_tangent, SymbolicZero):
            raise NotImplementedError('JVP for energy grid is not implemented')

        primals_out = fn(egrid, params)

        tvals, _ = jax.tree.flatten(params_tangent)
        if any(jnp.shape(v) != () for v in tvals):
            raise NotImplementedError(
                'JVP for non-scalar parameter is not implemented'
            )

        non_zero_tangents = [not isinstance(v, SymbolicZero) for v in tvals]
        idx = [i for i, v in enumerate(non_zero_tangents) if v]
        idx_arr = jnp.array(idx)
        nbatch = sum(non_zero_tangents)
        nparam = len(tvals)
        params_ravel, revert = ravel_pytree(params)
        free_params_values = params_ravel[idx_arr]
        free_params_abs = jnp.where(
            jnp.equal(free_params_values, 0.0),
            jnp.ones_like(free_params_values),
            jnp.abs(free_params_values),
        )
        free_params_abs = jnp.expand_dims(free_params_abs, axis=-1)
        row_idx = jnp.arange(nbatch)
        perturb_idx = jnp.zeros((nbatch, nparam)).at[row_idx, idx_arr].set(1.0)
        params_batch = jnp.full((nbatch, nparam), params_ravel)

        eps = jnp.finfo(egrid.dtype).eps
        f_vmap = jax.vmap(fn, in_axes=(None, 0), out_axes=0)
        revert = jax.vmap(revert, in_axes=0, out_axes=0)

        # See Numerical Recipes Chapter 5.7
        if method == 'central':
            perturb = free_params_abs * eps ** (1.0 / 3.0)
            params_pos_perturb = revert(params_batch + perturb_idx * perturb)
            out_pos_perturb = f_vmap(egrid, params_pos_perturb)
            params_neg_perturb = revert(params_batch - perturb_idx * perturb)
            out_neg_perturb = f_vmap(egrid, params_neg_perturb)
            d_out = (out_pos_perturb - out_neg_perturb) / (2.0 * perturb)
        else:
            perturb = free_params_abs * jnp.sqrt(eps)
            params_perturb = revert(params_batch + perturb_idx * perturb)
            out_perturb = f_vmap(egrid, params_perturb)
            d_out = (out_perturb - primals_out) / perturb

        free_params_tangent = jnp.array([tvals[i] for i in idx])
        tangents_out = free_params_tangent @ d_out
        return primals_out, tangents_out

    fn = jax.custom_jvp(fn)
    fn.defjvp(fdjvp, symbolic_zeros=True)

    fn = jax.custom_vjp(fn)

    return fn
