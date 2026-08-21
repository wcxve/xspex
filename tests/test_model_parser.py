from __future__ import annotations

from pathlib import Path

import pytest

from xspex._xspec import model_parser

_PARAMETER = 'index " " 1.0 0.0 0.0 10.0 10.0 0.1'


def _parse_model(tmp_path: Path, settings: str = ''):
    model_file = tmp_path / 'model.dat'
    model_file.write_text(
        f'test 1 0. 1.e20 C_test add 0{settings}\n{_PARAMETER}\n',
        encoding='utf-8',
    )
    return model_parser.get_models_info(
        model_file.as_posix(), parse_desc_and_link=False
    )['test']


def test_get_spectral_path_accepts_matching_link_created_by_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    headas = tmp_path / 'headas'
    alt_path = headas / 'spectral'
    alt_path.mkdir(parents=True)
    spectral_path = (tmp_path / 'spectral').resolve()
    original_symlink_to = Path.symlink_to

    def peer_create_link(
        path: Path,
        target: Path,
        target_is_directory: bool = False,
    ):
        original_symlink_to(
            path,
            target,
            target_is_directory=target_is_directory,
        )
        raise FileExistsError

    monkeypatch.setenv('HEADAS', headas.as_posix())
    monkeypatch.setattr(Path, 'symlink_to', peer_create_link)

    assert model_parser.get_spectral_path() == spectral_path.as_posix()


def test_get_spectral_path_rejects_wrong_link_created_by_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    headas = tmp_path / 'headas'
    alt_path = headas / 'spectral'
    alt_path.mkdir(parents=True)
    wrong_path = tmp_path / 'wrong-spectral'
    wrong_path.mkdir()
    original_symlink_to = Path.symlink_to

    def peer_create_wrong_link(
        path: Path,
        target: Path,
        target_is_directory: bool = False,
    ):
        original_symlink_to(
            path,
            wrong_path,
            target_is_directory=target_is_directory,
        )
        raise FileExistsError

    monkeypatch.setenv('HEADAS', headas.as_posix())
    monkeypatch.setattr(Path, 'symlink_to', peer_create_wrong_link)

    with pytest.raises(FileExistsError):
        model_parser.get_spectral_path()


def test_xspec_13_0_0_help_category_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spectral_path = tmp_path / 'spectral'
    html_path = spectral_path / 'help' / 'html'
    html_path.mkdir(parents=True)

    categories = {
        'Additive': ('Additive Model Components', 'testadd'),
        'Multiplicative': ('Multiplicative Model Components', 'test_mul'),
        'Convolution': ('Convolution Model Components', 'testcon'),
        'Pileup': ('Pile-Up Model Components', 'testpile'),
        'Mixing': ('Mixing Model Components', 'testmix'),
    }
    contents_links = []
    for index, (mtype, (title, model)) in enumerate(categories.items()):
        # XSPEC 13.0.0 conda packages can point these aliases at unrelated
        # pages that contain no ChildLinks list.
        (html_path / f'{mtype}.html').write_text(
            '<html><title>Unrelated page</title></html>',
            encoding='utf-8',
        )
        category_file = f'node{index + 100}.html'
        category_links = (
            '<ul class="ChildLinks">'
            f'<li><a href="{model}.html">{model}: {mtype} model</a></li>'
        )
        if mtype == 'Mixing':
            category_links += (
                '<li><a href="section.html">'
                'Failure modes and regularization</a></li>'
            )
        (html_path / category_file).write_text(
            category_links + '</ul>',
            encoding='utf-8',
        )
        contents_links.append(f'<a href="{category_file}">{title}</a>')

    (html_path / 'node1.html').write_text(
        ''.join(contents_links),
        encoding='utf-8',
    )
    monkeypatch.setattr(
        model_parser,
        'get_spectral_path',
        lambda: spectral_path.as_posix(),
    )

    model_info = model_parser.get_models_desc_and_link()

    for mtype, (_, model) in categories.items():
        model_key = model.replace('_', '')
        assert model_info[model_key]['desc'] == f'{mtype} model.'
    assert 'test_mul' not in model_info
    assert 'failure modes and regularization' not in model_info


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
