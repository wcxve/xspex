from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from xspex._compiled.lib.libxspex import get_xla_ffi_handler, xspec_version
from xspex._fdjvp import define_fdjvp
from xspex._xspec.model_parser import get_models_info
from xspex._xspec.types import XspecModelType

if TYPE_CHECKING:
    from collections.abc import Callable

    from xspex._xspec.types import XspecModel

__all__ = ['get_model', 'list_models']

FFI_TARGET_NAME_TEMPLATE = 'xspex_{name}'
MODELS_INFO: dict[str, XspecModel] = get_models_info()
MODELS: dict[str, Callable] = {}
SUPPORTED_TYPES = (XspecModelType.Add, XspecModelType.Mul, XspecModelType.Con)
XSPEC_VERSION: str = xspec_version()


def get_model(name: str) -> tuple[Callable, XspecModel]:
    """Get a XSPEC model function by name.

    Parameters
    ----------
    name : str
        The name of the XSPEC model.

    Returns
    -------
    fn : callable
        The XSPEC model function.
    model : XspecModel
        A dataclass containing XSPEC model information.
    """
    name_ = name.casefold()
    add_model_fn_to_cache(name_)
    return MODELS[name_], MODELS_INFO[name_]


def list_models(mtype: str | None = None) -> list[str]:
    """List XSPEC models, optionally filtered by type.

    Parameters
    ----------
    mtype : str, optional
        The type of XSPEC models to list.

            - ``'add'``: additive XSPEC models.
            - ``'mul'``: multiplicative XSPEC models.
            - ``'con'``: convolution XSPEC models.

        The default is ``None``, which means all XSPEC models.

    Returns
    -------
    list of str
        A list of XSPEC model names.
    """
    if mtype is None:
        models = []
        for t in SUPPORTED_TYPES:
            models.extend(list_models(t.name.lower()))
        return models

    mtype_ = XspecModelType.from_str(mtype)
    if mtype_ not in SUPPORTED_TYPES:
        raise NotImplementedError(f'{mtype} model is not supported yet')

    return sorted(name for name, m in MODELS_INFO.items() if m.type == mtype_)


def add_model_fn_to_cache(name: str) -> None:
    name = name.casefold()

    if name in MODELS:
        return

    if name not in MODELS_INFO:
        raise ValueError(f'XSPEC v{XSPEC_VERSION} has no model named {name}')

    model_info = MODELS_INFO[name]

    if model_info.type == XspecModelType.Add:
        mtype = 'additive'
    elif model_info.type == XspecModelType.Mul:
        mtype = 'multiplicative'
    elif model_info.type == XspecModelType.Con:
        mtype = 'convolution'
    else:
        mtype = model_info.type.name.lower()
        raise NotImplementedError(f'{mtype} model {name} is not supported yet')

    jax.ffi.register_ffi_target(
        name=FFI_TARGET_NAME_TEMPLATE.format(name=name),
        fn=get_xla_ffi_handler(model_info.name),
        platform='cpu',
    )
    fn = generate_model_fn(
        name=name,
        n_params=model_info.n_params,
        is_convolution_model=model_info.type == XspecModelType.Con,
    )
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = (
        f'XSPEC {mtype} model `{name} <{model_info.link}>`_: {model_info.desc}'
    )
    fn = jax.jit(fn, static_argnames='init_string')
    fn = define_fdjvp(fn, model_info)
    MODELS[name] = fn


def generate_model_fn(
    name: str,
    n_params: int,
    is_convolution_model: bool,
) -> Callable[..., jax.Array]:
    def fn_full_sigs(
        params: jax.Array,
        egrid: jax.Array,
        input_model: jax.Array | None = None,
        spec_num: int | jax.Array = 1,
        init_string: str = '',
    ) -> jax.Array:
        if isinstance(spec_num, int):
            spec_num = jnp.array(spec_num, dtype=jnp.int64)
        else:
            spec_num = jnp.asarray(spec_num)

        check_input(
            name,
            n_params,
            params,
            egrid,
            spec_num,
            input_model,
            init_string,
        )

        call = jax.ffi.ffi_call(
            target_name=FFI_TARGET_NAME_TEMPLATE.format(name=name),
            result_shape_dtypes=jax.ShapeDtypeStruct(
                shape=(len(egrid) - 1,),
                dtype=jnp.float64,
            ),
            vmap_method='sequential',
        )

        if not is_convolution_model:
            return call(
                params,
                egrid,
                spec_num,
                init_string=init_string,
                skip_check=True,
            )
        else:
            return call(
                params,
                egrid,
                input_model,
                spec_num,
                init_string=init_string,
                skip_check=True,
            )

    if is_convolution_model:

        def con_model_fn(
            params: jax.Array,
            egrid: jax.Array,
            input_model: jax.Array,
            spec_num: int | jax.Array = 1,
            init_string: str = '',
        ):
            return fn_full_sigs(
                params=params,
                egrid=egrid,
                input_model=input_model,
                spec_num=spec_num,
                init_string=init_string,
            )

        return con_model_fn

    else:

        def model_fn(
            params: jax.Array,
            egrid: jax.Array,
            spec_num: int | jax.Array = 1,
            init_string: str = '',
        ):
            return fn_full_sigs(
                params=params,
                egrid=egrid,
                spec_num=spec_num,
                init_string=init_string,
            )

        return model_fn


def check_input(
    model_name: str,
    n_params: int,
    params: jax.Array,
    egrid: jax.Array,
    spec_num: jax.Array,
    input_model: jax.Array | None,
    init_string: str,
):
    if not (
        jnp.ndim(params) == 1
        and len(params) == n_params
        and jnp.dtype(params) == jnp.float64
    ):
        shape = jnp.shape(params)
        dtype = jnp.dtype(params)
        raise ValueError(
            f"XSPEC {model_name} model's params must be a 1-D array of "
            f'shape ({n_params},) and dtype float64, got shape {shape} '
            f'and dtype {dtype}'
        )

    if not (
        jnp.ndim(egrid) == 1
        and len(egrid) >= 2
        and jnp.dtype(egrid) == jnp.float64
    ):
        shape = egrid.shape
        dtype = egrid.dtype
        raise ValueError(
            f"XSPEC {model_name} model's egrid must be a 1-D array of "
            f'length >= 2 and dtype float64, got shape {shape} and '
            f'dtype {dtype}'
        )

    if spec_num is not None:
        if not (jnp.isscalar(spec_num) and jnp.dtype(spec_num) == jnp.int64):
            shape = spec_num.shape
            dtype = spec_num.dtype
            raise ValueError(
                f"XSPEC {model_name} model's spec_num must be a scalar of "
                f'dtype int64, got shape {shape} and dtype {dtype}'
            )

    if input_model is not None:
        if not (
            jnp.ndim(input_model) == 1
            and jnp.dtype(input_model) == jnp.float64
        ):
            shape = input_model.shape
            dtype = input_model.dtype
            raise ValueError(
                f"XSPEC {model_name} model's input_model must be a 1-D "
                f'array of dtype float64, got shape {shape} and dtype '
                f'{dtype}'
            )
        if len(egrid) != len(input_model) + 1:
            raise ValueError(
                f'egrid size ({egrid.shape[-1]}) and input_model size '
                f'({input_model.shape[-1]}) are not consistent'
            )

    if not isinstance(init_string, str):
        raise ValueError(
            f"XSPEC {model_name} model's init_string must be a string, "
            f'got type {type(init_string)}'
        )
