from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from xspex._compiled.lib.libxspex import xla_ffi_handlers
from xspex._xspec.model_parser import get_models_info
from xspex._xspec.types import XspecModelType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from xspex._xspec.types import XspecModel

__all__ = ['get_model', 'list_models']

# Only include additive, multiplicative, and convolution models
_MODELS_INFO: dict[str, XspecModel] = {
    k: v
    for k, v in get_models_info().items()
    if v.type in {XspecModelType.Add, XspecModelType.Mul, XspecModelType.Con}
}
_MODELS: dict[str, Callable] = {}
_XLA_FFI_HANDLERS = xla_ffi_handlers()


def get_model(name: str) -> tuple[Callable, XspecModel]:
    """Get a XSPEC model function by name.

    Parameters
    ----------
    name : str
        The name of the XSPEC model.

    Returns
    -------
    model : callable
        The XSPEC model function.
    info : XspecModel
        The XSPEC model information.
    """
    name_ = name.casefold()
    _add_model_fn_to_cache(name_)
    return _MODELS[name_], _MODELS_INFO[name_]


def list_models(
    mtype: Literal['all', 'add', 'mul', 'con'] = 'all',
) -> list[str]:
    """List XSPEC models, optionally filtered by type.

    Parameters
    ----------
    mtype : {'all', 'add', 'mul', 'con'}, optional
        The type of XSPEC models to list.

            - ``'all'``: all XSPEC models.
            - ``'add'``: additive XSPEC models.
            - ``'mul'``: multiplicative XSPEC models.
            - ``'con'``: convolution XSPEC models.

        The default is ``'all'``.

    Returns
    -------
    list of str
        A list of XSPEC model names.
    """
    if mtype == 'all':
        return list(_MODELS_INFO.keys())

    if mtype == 'add':
        mtype_ = XspecModelType.Add
    elif mtype == 'mul':
        mtype_ = XspecModelType.Mul
    elif mtype == 'con':
        mtype_ = XspecModelType.Con
    else:
        raise ValueError(f'invalid model type: {mtype}')

    return [
        name for name, model in _MODELS_INFO.items() if model.type == mtype_
    ]


def _add_model_fn_to_cache(name: str) -> None:
    name = name.casefold()

    if name in _MODELS:
        return

    if name not in _MODELS_INFO:
        raise ValueError(f'no valid XSPEC model named {name}')

    model_info = _MODELS_INFO[name]

    if model_info.type == XspecModelType.Add:
        mtype = 'additive'
        fn = _generate_model_fn(name, model_info.n_params)
    elif model_info.type == XspecModelType.Mul:
        mtype = 'multiplicative'
        fn = _generate_model_fn(name, model_info.n_params)
    elif model_info.type == XspecModelType.Con:
        mtype = 'convolution'
        fn = _generate_con_model_fn(name, model_info.n_params)
    else:
        mtype = model_info.type.name.lower()
        raise ValueError(f'{mtype} model {name} is not supported yet')

    jax.ffi.register_ffi_target(
        name,
        _XLA_FFI_HANDLERS[model_info.name],
        platform='cpu',
    )
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = (
        f'XSPEC {mtype} model `{name} <{model_info.link}>`_: {model_info.desc}'
    )
    _MODELS[name] = jax.jit(fn, static_argnames='init_string')


def _check_input(n_params, params, egrid, spec_num=None, model=None):
    if not (
        jnp.ndim(params) == 1
        and len(params) == n_params
        and jnp.dtype(params) == jnp.float64
    ):
        raise ValueError(
            f'params must be a 1-D array of shape ({n_params},) '
            f'and dtype float64, got shape {jnp.shape(params)} '
            f'and dtype {jnp.dtype(params)}'
        )

    if not (jnp.ndim(egrid) == 1 and jnp.dtype(egrid) == jnp.float64):
        raise ValueError(
            'egrid must be a 1-D array of dtype float64, got '
            f'shape {jnp.shape(egrid)} and dtype {jnp.dtype(egrid)}'
        )

    if spec_num is not None:
        if not (jnp.isscalar(spec_num) and jnp.dtype(spec_num) == jnp.int64):
            raise ValueError(
                'spec_num must be a scalar of dtype int64, got '
                f'{spec_num} and dtype {jnp.dtype(spec_num)}'
            )

    if model is not None:
        if not (jnp.ndim(model) == 1 and jnp.dtype(model) == jnp.float64):
            raise ValueError(
                'model must be a 1-D array with dtype float64, got '
                f'shape={jnp.shape(model)} and dtype={jnp.dtype(model)}'
            )
        if len(egrid) != len(model) + 1:
            raise ValueError(
                f'egrid size ({len(egrid)}) and model size ({len(model)}) '
                'are not consistent'
            )


def _generate_model_fn(name: str, n_params: int):
    def fn(params, egrid, spec_num, *, init_string=''):
        _check_input(n_params, params, egrid, spec_num)
        return jax.ffi.ffi_call(
            target_name=name,
            result_shape_dtypes=jax.ShapeDtypeStruct(
                shape=(len(egrid) - 1,),
                dtype=jnp.float64,
            ),
            vmap_method='sequential',
        )(
            params,
            egrid,
            spec_num,
            init_string=init_string,
            skip_check=True,
        )

    return fn


def _generate_con_model_fn(name: str, n_params: int):
    def fn(params, egrid, model, spec_num, *, init_string=''):
        _check_input(n_params, params, egrid, spec_num, model)
        return jax.ffi.ffi_call(
            target_name=name,
            result_shape_dtypes=jax.ShapeDtypeStruct(
                shape=(len(egrid) - 1,),
                dtype=jnp.float64,
            ),
            vmap_method='sequential',
        )(
            params,
            egrid,
            model,
            spec_num,
            init_string=init_string,
            skip_check=True,
        )

    return fn
