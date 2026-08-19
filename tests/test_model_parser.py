from __future__ import annotations

from pathlib import Path

import pytest

from xspex._xspec.model_parser import get_models_info

_PARAMETER = 'index " " 1.0 0.0 0.0 10.0 10.0 0.1'


def _parse_model(tmp_path: Path, settings: str = ''):
    model_file = tmp_path / 'model.dat'
    model_file.write_text(
        f'test 1 0. 1.e20 C_test add 0{settings}\n{_PARAMETER}\n',
        encoding='utf-8',
    )
    return get_models_info(model_file.as_posix(), parse_desc_and_link=False)[
        'test'
    ]


@pytest.mark.parametrize(
    'gradient',
    [
        'grad=g',
        'grad=gv',
        'grad=gv:testGradient,testVJP',
    ],
)
def test_parse_xspec_13_0_0_gradient_setting(tmp_path: Path, gradient: str):
    model = _parse_model(tmp_path, f' {gradient}')

    assert model.data_depend is False
    assert model.init_string == ''


@pytest.mark.parametrize(
    ('data_depend', 'expected'),
    [('0', False), ('1', True)],
)
def test_xspec_13_0_0_gradient_setting_after_data_depend(
    tmp_path: Path,
    data_depend: str,
    expected: bool,
):
    model = _parse_model(tmp_path, f' {data_depend} grad=gv')

    assert model.data_depend is expected
    assert model.init_string == ''


def test_parse_legacy_model_settings(tmp_path: Path):
    with pytest.warns(UserWarning, match='extra string is not supported'):
        model = _parse_model(tmp_path, ' 1 init extra value')

    assert model.data_depend is True
    assert model.init_string == 'init'


def test_reject_unknown_model_setting(tmp_path: Path):
    with pytest.raises(ValueError, match='invalid model definition'):
        _parse_model(tmp_path, ' unknown=value')
