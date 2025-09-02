from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from xspex._xspec.types import XspecModelType, XspecParamType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import ArrayLike

    from xspex._xspec.types import XspecModel

JNP_AT_KWS = {
    'indices_are_sorted': True,
    'unique_indices': True,
    'mode': 'promise_in_bounds',
}


def define_fdjvp(
    fn: Callable,
    model_info: XspecModel,
    fixed: ArrayLike | None = None,
    delta: float = 0.0,
    method: Literal['central', 'forward'] = 'central',
) -> Callable:
    """Define the JVP using finite difference approximation.

    .. note::
        The JVP rule for `input_model` of convolutional models is not
        implemented.

    Parameters
    ----------
    fn: callable
        XSPEC model function.
    model_info: XspecModel, optional
        XSPEC model information.
    fixed: array_like, optional
        Boolean array indicating which parameters are fixed. The parameters
        with ``True`` in `fixed` are fixed during the computation of finite
        differences. The array must have the same size as parameters.
        The default is ``None``, which means use fixed flags in `model_info`.
    delta: float, optional
        Set the step size used in finite differences.

            - If `delta` > 0, the step size used in finite differences
              is computed as ``delta * abs(param_value)``.
            - If `delta` == 0, the step size used in finite differences
              is computed as ``d * abs(param_value)``, where ``d = eps ** a``,
              ``eps`` is the numerical precision of model function output,
              and ``a=1/3`` if ``method='central'`` and ``a=1/2`` if
              ``method='forward'``.
            - If `delta` < 0, use the step size in `model_info`.

        The default is ``0.0``. Also note that in XSPEC, the default value of
        `delta` is ``0.01``.
    method: {'central', 'forward'}
        Method for finite differences. Available options are:

            - ``'central'``: central finite differences
            - ``'forward'``: forward finite differences

        The default is ``'central'``.

    Returns
    -------
    Callable
        Function with JVP defined.
    """
    if method not in {'central', 'forward'}:
        raise ValueError("method must be 'central' or 'forward'")

    model_name = model_info.name
    n_params = model_info.n_params

    # fixed and deltas for finite differences
    p_info = model_info.parameters
    fixed_types = (XspecParamType.Scale, XspecParamType.Switch)
    must_fixed = np.array([p.type in fixed_types for p in p_info], dtype=bool)
    fixed_default = np.array([p.fixed for p in p_info], dtype=bool)
    if fixed is None:
        fixed = fixed_default
    else:
        fixed = np.array(fixed, dtype=bool)
        if fixed.size != n_params:
            raise ValueError('`fixed` must have same size as parameters')
        error_fixed = must_fixed & (~fixed)
        if np.any(error_fixed):
            names = [p_info[i].name for i in np.flatnonzero(error_fixed)]
            raise ValueError(
                f'XSPEC model {model_name} parameters {names} must be fixed'
            )

    delta = float(delta)
    if not np.isfinite(delta):
        raise ValueError('`delta` must be finite and not NaN')

    deltas_default = np.array([p.delta or 0.0 for p in p_info])

    if delta >= 0:
        use_rel_step = True
        if delta == 0:
            pow_idx = 1.0 / 3.0 if method == 'central' else 0.5
            delta = model_info.eps**pow_idx
        deltas = np.full(n_params, delta)
    else:
        use_rel_step = False
        deltas = deltas_default

    if model_info.type != XspecModelType.Con:
        # For additive and multiplicative models, there are 4 arguments:
        #     params, egrid, spec_num, init_string
        nondiff_argnums = (1, 2, 3)
    else:
        # For convolutional models, there are 5 arguments:
        #     params, egrid, input_model, spec_num, init_string
        nondiff_argnums = (1, 3, 4)

    fn = jax.custom_jvp(fn, nondiff_argnums=nondiff_argnums)

    @partial(fn.defjvp, symbolic_zeros=True)
    def _(*args):
        *others, primals, tangents = args

        if model_info.type != XspecModelType.Con:
            (params,) = primals
            (params_tangent,) = tangents
        else:
            egrid, spec_num, init_string = others
            (params, input_model) = primals
            (params_tangent, input_model_tangent) = tangents
            others = (egrid, input_model, spec_num, init_string)
            if not isinstance(input_model_tangent, SymbolicZero):
                raise NotImplementedError(
                    f"JVP rule for XSPEC {model_name} model's input_model "
                    'is not implemented'
                )

        primal_out = fn(params, *others)

        # mask of params with zero tangents input
        zero_tangent_mask = np.array(
            [isinstance(t, SymbolicZero) for t in params_tangent],
            dtype=bool,
        )

        # mask and idx of params that does not contribute to JVP computation
        static_mask = np.logical_or(zero_tangent_mask, fixed)
        idx = np.flatnonzero(~static_mask)

        if not idx.size:
            tangent_out = jax.lax.zeros_like_array(primal_out)
            return primal_out, tangent_out

        if use_rel_step:
            params_free = params.at[idx].get(**JNP_AT_KWS)
            perturb = deltas[idx] * jnp.abs(params_free)
            perturb = jnp.where(perturb != 0.0, perturb, deltas_default[idx])
        else:
            perturb = deltas[idx]

        perturb_bc = perturb[:, None]  # shape=(n_batches, 1), for broadcasting

        fn_vmap = jax.vmap(lambda p: fn(p, *others))
        n_batches = idx.size
        batch_idx = jnp.arange(n_batches)
        params_batch = jnp.full((n_batches, n_params), params)
        params_batch_at_idx = params_batch.at[batch_idx, idx]
        if method == 'central':
            params_plus = params_batch_at_idx.add(perturb, **JNP_AT_KWS)
            params_minus = params_batch_at_idx.add(-perturb, **JNP_AT_KWS)
            primal_out_plus = fn_vmap(params_plus)
            primal_out_minus = fn_vmap(params_minus)
            jac = (primal_out_plus - primal_out_minus) / (2.0 * perturb_bc)
        else:  # forward
            params_plus = params_batch_at_idx.add(perturb, **JNP_AT_KWS)
            primal_out_plus = fn_vmap(params_plus)
            jac = (primal_out_plus - primal_out) / perturb_bc

        free_params_tangent = jnp.array([params_tangent[i] for i in idx])
        tangents_out = free_params_tangent @ jac
        return primal_out, tangents_out

    return fn
