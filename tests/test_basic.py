from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from xspec import Xset

import xspex as xx


def test_xspec_version():
    assert xx.xspec_version() == Xset.version[1]


def test_abund():
    assert xx.abund() == Xset.abund[:4]


def test_xsect():
    assert xx.xsect() == Xset.xsect


def test_cosmo():
    keys = ['H0', 'q0', 'lambda0']
    xx_cosmo = xx.cosmo()
    xs_cosmo = Xset.cosmo
    for i, key in enumerate(keys):
        assert_allclose(
            np.float32(xx_cosmo[key]),
            np.float32(xs_cosmo[i]),
            err_msg=key,
        )


def test_list_models():
    models_all = xx.list_models()
    models_add = xx.list_models('add')
    models_mul = xx.list_models('mul')
    models_con = xx.list_models('con')
    assert models_all == models_add + models_mul + models_con
    assert len(set(models_all)) == len(models_all)

    # Not supported models
    with pytest.raises(NotImplementedError):
        xx.list_models('mix')
    with pytest.raises(NotImplementedError):
        xx.list_models('acn')
    with pytest.raises(NotImplementedError):
        xx.list_models('amx')

    # Invalid input
    with pytest.raises(ValueError):
        xx.list_models('invalid')


def test_get_model():
    # Singleton model function and model info
    for name in xx.list_models():
        fn1, info1 = xx.get_model(name)
        fn2, info2 = xx.get_model(name)
        assert fn1 is fn2
        assert info1 is info2

    # Invalid input
    with pytest.raises(ValueError):
        xx.get_model('invalid')

    # Unsupported models
    with pytest.raises(NotImplementedError):
        xx.get_model('pileup')
