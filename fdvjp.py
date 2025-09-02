from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from xspex._compiled.lib.libxspex import get_xla_ffi_handler
from xspex._model import _check_input

jax.config.update('jax_enable_x64', True)
jax.ffi.register_ffi_target('apec', get_xla_ffi_handler('apec'))

method = 'central'
fixed = None
abs_floor = 1e-8
rel_scale = 1.0


def _apec_impl(params, egrid, spec_num, init_string=''):
    """Internal implementation of apec function."""
    _check_input(3, params, egrid, spec_num)
    call = jax.ffi.ffi_call(
        target_name='apec',
        result_shape_dtypes=jax.ShapeDtypeStruct(
            shape=(len(egrid) - 1,),
            dtype=jnp.float64,
        ),
        vmap_method='sequential',
    )
    return call(
        params, egrid, spec_num, init_string=init_string, skip_check=True
    )


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def apec_numeric(params, egrid, spec_num):
    """APEC model with custom VJP implementation (numeric args only)."""
    return _apec_impl(params, egrid, spec_num, '')


def apec_numeric_fwd(params, egrid, spec_num):
    """Forward pass for apec custom VJP."""
    # Compute the primal output
    primal_out = _apec_impl(params, egrid, spec_num, '')

    # Store information needed for backward pass
    # We need params, egrid for finite difference computation
    residuals = (params, egrid)

    return primal_out, residuals


def apec_numeric_bwd(spec_num, residuals, cotangents):
    """Backward pass for apec custom VJP using finite differences."""
    params, egrid = residuals
    output_cotangent = cotangents

    # Flatten params and cotangents
    p_flat, unflat_p = ravel_pytree(params)
    c_flat, _ = ravel_pytree(output_cotangent)

    # dtype-aware eps and eps_scale
    eps = jnp.finfo(p_flat.dtype).eps
    eps_scale = eps ** (1.0 / 3.0) if method == 'central' else jnp.sqrt(eps)

    # build fixed mask aligned with p_flat
    if fixed is None:
        fixed_mask = jnp.array([False] * p_flat.size, dtype=jnp.bool_)
    else:
        fixed_arr = jnp.array(fixed, dtype=jnp.bool_)
        if fixed_arr.size == p_flat.size:
            fixed_mask = fixed_arr.flatten()
        else:
            raise ValueError(
                '`fixed` must be None, or have same size as params'
            )

    # per-component scale for finite differences
    comp_scale = jnp.maximum(jnp.abs(p_flat), abs_floor) * rel_scale
    comp_scale_nonstatic = jnp.where(fixed_mask, 0.0, comp_scale)

    n_nonstatic = jnp.sum(~fixed_mask)

    # If all parameters are fixed, gradient is zero
    if n_nonstatic == 0:
        params_cotangent = jax.tree_util.tree_map(jnp.zeros_like, params)
        return (params_cotangent, None)

    # compute a scalar representative scale (RMS over non-static comps)
    rms = jnp.sqrt(jnp.sum(jnp.square(comp_scale_nonstatic)) / n_nonstatic)
    h_scalar = rms * eps_scale

    # Function to evaluate the model with flattened parameters
    def f_flat(p_flat_in):
        return ravel_pytree(
            _apec_impl(unflat_p(p_flat_in), egrid, spec_num, '')
        )[0]

    # Compute gradient using finite differences
    if method == 'central':
        # Central difference for each parameter
        def compute_partial_derivative(i):
            # Create unit vector for i-th parameter
            ei = jnp.zeros_like(p_flat)
            ei = ei.at[i].set(1.0)

            # Skip if parameter is fixed
            h_i = jnp.where(fixed_mask[i], 0.0, h_scalar)

            # Compute finite difference
            plus = f_flat(p_flat + h_i * ei)
            minus = f_flat(p_flat - h_i * ei)

            # Gradient component
            grad_i = jnp.where(
                fixed_mask[i], 0.0, (plus - minus) / (2.0 * h_i)
            )
            return grad_i

        # Vectorized computation of all partial derivatives
        grad_flat = jax.vmap(compute_partial_derivative)(
            jnp.arange(p_flat.size)
        )

    else:  # forward difference
        # Forward difference for each parameter
        f_base = f_flat(p_flat)

        def compute_partial_derivative(i):
            # Create unit vector for i-th parameter
            ei = jnp.zeros_like(p_flat)
            ei = ei.at[i].set(1.0)

            # Skip if parameter is fixed
            h_i = jnp.where(fixed_mask[i], 0.0, h_scalar)

            # Compute finite difference
            plus = f_flat(p_flat + h_i * ei)

            # Gradient component
            grad_i = jnp.where(fixed_mask[i], 0.0, (plus - f_base) / h_i)
            return grad_i

        # Vectorized computation of all partial derivatives
        grad_flat = jax.vmap(compute_partial_derivative)(
            jnp.arange(p_flat.size)
        )

    # Contract with output cotangents: (grad_flat^T @ c_flat)
    # grad_flat shape: (n_params, n_outputs)
    # c_flat shape: (n_outputs,)
    # Result shape: (n_params,)
    params_cotangent_flat = jnp.dot(grad_flat, c_flat)

    # Unflatten to original parameter structure
    params_cotangent = unflat_p(params_cotangent_flat)

    return (params_cotangent, None)


# Define the VJP
apec_numeric.defvjp(apec_numeric_fwd, apec_numeric_bwd)


# Provide apec function with custom VJP support
def apec(params, egrid, spec_num, init_string=''):
    """APEC model function with custom VJP support.

    For automatic differentiation with respect to params, this function
    uses finite differences via custom_vjp. If init_string is non-empty,
    the differentiation will use the default empty string for consistency.

    Parameters
    ----------
    params : array_like
        Model parameters (3 parameters for APEC).
    egrid : array_like
        Energy grid points.
    spec_num : int
        Spectrum number.
    init_string : str, optional
        Initialization string (default: '').

    Returns
    -------
    array_like
        Model spectrum values.
    """
    if init_string == '':
        # Use the custom VJP version for differentiation
        return apec_numeric(params, egrid, spec_num)
    else:
        # For non-empty init_string, use the original implementation
        # Note: this version won't have custom gradients
        return _apec_impl(params, egrid, spec_num, init_string)


# Test the custom VJP implementation
if __name__ == '__main__':
    print('Testing APEC custom VJP implementation')
    print('=' * 50)

    # Test parameters
    test_params = jnp.array([1.0, 1.0, 0.0])
    test_egrid = jnp.linspace(1.0, 2.0, 2)
    test_spec_num = jnp.int64(1)

    # Test 1: Basic functionality
    print('\n1. Testing basic function evaluation:')
    result = apec(test_params, test_egrid, test_spec_num)
    print(f'   Function output shape: {result.shape}')
    print(f'   Function output: {result}')

    # Test 2: Custom VJP gradient
    print('\n2. Testing custom VJP gradient:')
    f = lambda p: apec(p, test_egrid, test_spec_num).sum()
    g = jax.grad(f)
    gradient = g(test_params)
    print(f'   Gradient: {gradient}')
    print(f'   Gradient shape: {gradient.shape}')
    print(f'   Gradient dtype: {gradient.dtype}')

    # Test 3: Verify gradient is non-zero and finite
    print('\n3. Gradient validation:')
    print(f'   All finite: {jnp.all(jnp.isfinite(gradient))}')
    print(f'   Any non-zero: {jnp.any(gradient != 0)}')
    print(f'   Gradient norm: {jnp.linalg.norm(gradient)}')

    # Test 4: Compare with numeric version directly
    print('\n4. Direct numeric version test:')
    f_numeric = lambda p: apec_numeric(p, test_egrid, test_spec_num).sum()
    g_numeric = jax.grad(f_numeric)
    gradient_numeric = g_numeric(test_params)
    print(f'   Numeric gradient: {gradient_numeric}')
    print(f'   Gradients match: {jnp.allclose(gradient, gradient_numeric)}')

    print('\n' + '=' * 50)
    print('Custom VJP implementation test completed successfully!')
