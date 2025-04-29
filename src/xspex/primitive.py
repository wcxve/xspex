"""Xspec model primitives for JAX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# Primitive is moved from jax.core to jax.extend.core in jax>=0.6.0
try:
    from jax.extend.core import Primitive
except ImportError:
    from jax.core import Primitive

# register_custom_call_target is replaced by register_ffi_target in jax>=0.6.0
try:
    from jax.ffi import register_ffi_target

    register_ffi_target = partial(register_ffi_target, api_version=0)
except ImportError:
    from jax.lib import xla_client

    register_ffi_target = xla_client.register_custom_call_target

import xspex


def avals_to_layouts(avals):
    return [list(reversed(range(aval.ndim))) for aval in avals]


class XspecPrimitiveBase(Primitive, ABC):
    def __init__(self, name: str):
        name = str(name)
        if name not in xspex.list_models():
            raise ValueError(f"Xspec v{xspex.version()} has no '{name}' model")

        super().__init__(f'XS{name}')
        self._model_name = name
        self._nparam = len(xspex.info(name).parameters)

        self.def_impl(partial(xla.apply_primitive, self))
        # self.def_impl(getattr(xspex, name))

        self.def_abstract_eval(self.get_abstract_eval())

        mlir.register_lowering(self, self.lowering, platform='cpu')

        batching.primitive_batchers[self] = self.batching

        ad.primitive_jvps[self] = self.jvp

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_abstract_eval(self):
        pass

    @abstractmethod
    def lowering(self, *args, **kwargs):
        pass

    @abstractmethod
    def batching(self, vector_arg_values, batch_axes):
        pass

    def jvp(self, primals, tangents):
        raise NotImplementedError(f'JVP is not implemented for {self}')


class XspecPrimitive(XspecPrimitiveBase):
    def __call__(self, params, egrid, spec_num):
        return self.bind(params, egrid, spec_num)

    def get_abstract_eval(self):
        name = self.name
        npar = self._nparam

        def abstract_eval(params, egrid, spec_num):
            params_shape = jnp.shape(params)
            if params_shape == () or params_shape[-1] != npar:
                raise ValueError(f'{name} params shape must be (..., {npar})')
            egrid_shape = jnp.shape(egrid)
            if len(egrid_shape) != 1:
                raise ValueError('egrid must be 1D array')
            spec_num_shape = jnp.shape(spec_num)
            if len(spec_num_shape):
                raise ValueError('spec_num must be a integer')

            shape = params_shape[:-1] + (egrid_shape[-1] - 1,)
            dtype = jax.dtypes.canonicalize_dtype(egrid.dtype)
            return ShapedArray(shape=shape, dtype=dtype)

        return abstract_eval

    def lowering(self, ctx, params, egrid, spec_num):
        egrid_type = mlir.ir.RankedTensorType(egrid.type)
        etype = egrid_type.element_type
        if isinstance(etype, mlir.ir.F32Type):
            call_target_name = f'{self._model_name}_f32'
        elif isinstance(etype, mlir.ir.F64Type):
            call_target_name = f'{self._model_name}_f64'
        else:
            raise NotImplementedError(f'unsupported dtype {etype}')
        out_shape = ctx.avals_out[0].shape
        out_type = mlir.ir.RankedTensorType.get(out_shape, etype)
        out_n = mlir.ir_constant(out_shape[-1])
        out_batch = mlir.ir_constant(prod(out_shape[:-1]))
        return mlir.custom_call(
            call_target_name,
            result_types=[out_type],
            operands=[params, egrid, spec_num, out_n, out_batch],
            operand_layouts=avals_to_layouts(ctx.avals_in) + [()] * 2,
            result_layouts=avals_to_layouts(ctx.avals_out),
        ).results

    def batching(self, vector_arg_values, batch_axes):
        if batch_axes[1] is not None:
            raise NotImplementedError('egrid batching is not implemented')
        if batch_axes[2] is not None:
            raise NotImplementedError('spec_num batching is not implemented')

        params, egrid, spec_num = vector_arg_values
        params = jnp.moveaxis(params, batch_axes[0], 0)
        return self(params, egrid, spec_num), 0

    def jvp(self, primals, tangents):
        params, egrid, spec_num = primals
        d_params = tangents[0]

        primals_out = self(params, egrid, spec_num)
        f_vmap = jax.vmap(jax.jit(self), in_axes=(0, None, None), out_axes=0)
        eps = jnp.finfo(params.dtype).eps
        identity = jnp.eye(len(params))
        params_abs = jnp.where(
            jnp.equal(params, 0.0), jnp.ones_like(params), jnp.abs(params)
        )
        params_abs = jnp.expand_dims(params_abs, axis=-1)

        # See Numerical Recipes Chapter 5.7
        USE_CENTRAL_DIFF = True
        if USE_CENTRAL_DIFF:
            delta = params_abs * eps ** (1.0 / 3.0)
            params_pos_perturb = params + identity * delta
            out_pos_perturb = f_vmap(params_pos_perturb, egrid, spec_num)
            params_neg_perturb = params - identity * delta
            out_neg_perturb = f_vmap(params_neg_perturb, egrid, spec_num)
            d_out = (out_pos_perturb - out_neg_perturb) / (2.0 * delta)
        else:
            delta = params_abs * jnp.sqrt(eps)
            params_perturb = params + identity * delta
            out_perturb = f_vmap(params_perturb, egrid, spec_num)
            d_out = (out_perturb - primals_out) / delta

        tangents_out = d_params @ d_out
        return primals_out, tangents_out


class XspecConvPrimitive(XspecPrimitiveBase):
    def __call__(self, params, egrid, flux, spec_num):
        return self.bind(params, egrid, flux, spec_num)

    def get_abstract_eval(self):
        name = self.name
        npar = self._nparam

        def abstract_eval(params, egrid, flux, spec_num):
            params_shape = jnp.shape(params)
            if params_shape == () or params_shape[-1] != npar:
                raise ValueError(f'{name} params shape must be (..., {npar})')
            egrid_shape = jnp.shape(egrid)
            if len(egrid_shape) != 1:
                raise ValueError('egrid must be 1D array')
            flux_shape = jnp.shape(flux)
            if flux_shape == () or flux_shape[-1] != egrid_shape[-1] - 1:
                raise ValueError('flux shape must be (..., len(egrid) - 1)')
            spec_num_shape = jnp.shape(spec_num)
            if len(spec_num_shape):
                raise ValueError('spec_num must be a integer')

            if len(params_shape) > 1 and len(flux_shape) > 1:
                if params_shape[:-1] != flux_shape[:-1]:
                    raise ValueError(
                        'params and flux should have the same batch size'
                    )
            if len(params_shape) >= len(flux_shape):
                shape = params_shape[:-1] + (egrid_shape[-1] - 1,)
            else:
                shape = flux_shape[:-1] + (egrid_shape[-1] - 1,)
            dtype = jax.dtypes.canonicalize_dtype(egrid.dtype)
            return ShapedArray(shape=shape, dtype=dtype)

        return abstract_eval

    def lowering(self, ctx, params, egrid, flux, spec_num):
        egrid_type = mlir.ir.RankedTensorType(egrid.type)
        etype = egrid_type.element_type
        if isinstance(etype, mlir.ir.F32Type):
            call_target_name = f'{self._model_name}_f32'
        elif isinstance(etype, mlir.ir.F64Type):
            call_target_name = f'{self._model_name}_f64'
        else:
            raise NotImplementedError(f'unsupported dtype {etype}')

        params_shape = ctx.avals_in[0].shape
        params_batch = mlir.ir_constant(prod(params_shape[:-1]))

        flux_shape = ctx.avals_in[2].shape
        flux_batch = mlir.ir_constant(prod(flux_shape[:-1]))

        out_shape = ctx.avals_out[0].shape
        out_type = mlir.ir.RankedTensorType.get(out_shape, etype)
        out_n = mlir.ir_constant(out_shape[-1])

        return mlir.custom_call(
            call_target_name,
            result_types=[out_type],
            operands=[
                params,
                egrid,
                flux,
                spec_num,
                out_n,
                params_batch,
                flux_batch,
            ],
            operand_layouts=avals_to_layouts(ctx.avals_in) + [()] * 3,
            result_layouts=avals_to_layouts(ctx.avals_out),
        ).results

    def batching(self, vector_arg_values, batch_axes):
        if batch_axes[1] is not None:
            raise NotImplementedError('egrid batching is not implemented')
        if batch_axes[3] is not None:
            raise NotImplementedError('spec_num batching is not implemented')

        params, egrid, flux, spec_num = vector_arg_values
        if batch_axes[0] is not None:
            params = jnp.moveaxis(params, batch_axes[0], 0)
        if batch_axes[2] is not None:
            flux = jnp.moveaxis(flux, batch_axes[2], 0)
        return self(params, egrid, flux, spec_num), 0


def get_primitive(
    model: str,
) -> tuple[XspecPrimitive | XspecConvPrimitive, xspex.XspecModel]:
    """Return the primitive for the given Xspec model.

    Parameters
    ----------
    model : str
        The Xspec model name.

    Returns
    -------
    primitive : XspecPrimitive or XspecConvPrimitive
        The primitive for the model.
    model : XspecModel
        The dataclass that describes the model.

    Examples
    --------

    >>> apec, info = get_primitive('apec')
    >>> apec
    'apec'

    """
    model = str(model)
    check = model.casefold()
    p = next(
        (v for k, v in XSModel['primitive'].items() if k.casefold() == check),
        None,
    )
    if p is None:
        raise ValueError(f"Unrecognized Xspec model '{model}'")

    return p, xspex.info(model)


for k, v in xspex.xla_registrations().items():
    register_ffi_target(k, v, platform='cpu')

add = xspex.list_models(xspex.ModelType.Add)
mul = xspex.list_models(xspex.ModelType.Mul)
con = xspex.list_models(xspex.ModelType.Con)
primitive = {m: XspecPrimitive(m) for m in add + mul}
primitive |= {m: XspecConvPrimitive(m) for m in con}
XSModel = {
    'add': {m: xspex.info(m) for m in add},
    'mul': {m: xspex.info(m) for m in mul},
    'con': {m: xspex.info(m) for m in con},
    'primitive': primitive,
}
del k, v, add, mul, con, primitive
