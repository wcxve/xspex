import numpy as np
import pytest

import xspex as x

# We need to set up the cosmology as this is currently not done in FNINIT.
#
x.cosmology(H0=70, lambda0=0.73, q0=0)

# We want to ensure we have a fixed abundance / cross section
# for the checks below. If we didn't set them here then it
# would depend on the user's ~/.xspec/Xspec.init file
#
x.abundance('lodd')
x.cross_section('vern')
version = x.version().split('.')
major = int(version[0])
minor = int(version[1])
if major > 12 or (major == 12 and minor >= 15):
    x.set_model_string('NEIVERS', '3.1.2')


def get_dtype(model):
    """What is the drtype used by this model?"""

    return {
        x.LanguageStyle.CppStyle8: np.float64,
        x.LanguageStyle.CStyle8: np.float64,
        x.LanguageStyle.F77Style4: np.float32,
        x.LanguageStyle.F77Style8: np.float64,
    }[model.language]


def test_have_version():
    """Minimal chekc of get_versino"""
    v = x.version()
    assert v != ''
    assert len(v.split('.')) == 3, v


def test_default_abundance():
    assert x.abundance() == 'lodd'


def test_default_cross_section():
    assert x.cross_section() == 'vern'


def test_number_elements():
    """Technically this could change in an XSPEC release"""
    assert x.number_elements == 30


@pytest.mark.parametrize(
    'z,name', [(1, 'H'), (2, 'He'), (17, 'Cl'), (29, 'Cu'), (30, 'Zn')]
)
def test_elementName(z, name):
    assert x.element_name(z) == name


@pytest.mark.parametrize(
    'z,value',
    [
        (1, 1.0),
        (2, 0.07919999957084656),
        (17, 1.8199999374246545e-07),
        (29, 1.8199999729517913e-08),
        (30, 4.2700001756657e-08),
    ],
)
def test_elementAbundance(z, value):
    x.abundance('lodd')
    assert x.element_abundance(z) == pytest.approx(value)
    assert x.element_abundance(x.element_name(z)) == pytest.approx(value)


def test_cosmo_get():
    """Check the values we set above"""
    ans = x.cosmology()
    assert ans == pytest.approx({'H0': 70, 'lambda0': 0.73, 'q0': 0.0})


def test_has_wabs_info():
    """Check wabs.

    This is used as the model.dat entry is not likely to change
    for it.
    """

    assert 'wabs' in x.list_models()
    assert 'wabs' in x.list_models(modeltype=x.ModelType.Mul)
    assert 'wabs' not in x.list_models(modeltype=x.ModelType.Add)
    assert 'wabs' not in x.list_models(modeltype=x.ModelType.Con)

    model = x.info('wabs')
    assert model.name == 'wabs'
    assert model.modeltype == x.ModelType.Mul
    assert len(model.parameters) == 1

    # Technically we should be able to support this if it changes,
    # but it indicates that test_eval_wabs and test_eval_wabs_inline
    # may need the reference array changed.
    #
    assert model.language == x.LanguageStyle.F77Style4

    par = model.parameters[0]
    assert par.name == 'nH'
    assert par.default == pytest.approx(1.0)
    assert not par.frozen


WABS_MODEL = [8.8266723e-05, 2.3582002e-02, 1.5197776e-01]


@pytest.mark.parametrize('pars', [[0.1], (0.1,), np.asarray([0.1])])
@pytest.mark.parametrize(
    'energies',
    [[0.1, 0.2, 0.3, 0.4], (0.1, 0.2, 0.3, 0.4), np.arange(0.1, 0.5, 0.1)],
)
def test_eval_wabs(pars, energies):
    """Explicit tests of a model.

    This checks a "random" model - chosen as wabs as it's assumed it's
    not going to change - evaluates as expected.

    """

    y = x.wabs(energies=energies, pars=pars)

    # Let's assume this isn't going to change much
    #
    assert y == pytest.approx(WABS_MODEL)


# Unfortunately some models need to be skipped for
# some reason. This is obviously going to be version-specific.
#
# grbjet occasional failures has been reported to XSPEC (in 12.12.0).
#
# rfxconv and xilconv require additinal setup (e.g. energy range) that I have
# no energy to diesntangle, so we skip
#
# rgsxsrc requires extra setup (an image file)
#
MODELS_ADD = x.list_models(modeltype=x.ModelType.Add)
MODELS_MUL = x.list_models(modeltype=x.ModelType.Mul)
MODELS_CON = x.list_models(modeltype=x.ModelType.Con)

MODELS_ADD_SKIP = ['grbjet']
MODELS_MUL_SKIP = []
MODELS_CON_SKIP = ['rfxconv', 'rgsext', 'rgsxsrc', 'xilconv']


@pytest.mark.parametrize('model', MODELS_ADD)
def test_eval_add(model):
    """Evaluate an additive model.

    Check we get a value > 0. To do so we ensure redshift > 0
    as this is an issue for some models.

    We skip models with can_cache set to False because we
    assume that the necessary setup has not been called
    """

    info = x.info(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_ADD_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    mfunc = getattr(x, model)

    pars = [
        0.1 if p.name.casefold() == 'redshift' else p.default
        for p in info.parameters
    ]

    egrid = np.arange(0.1, 10, 0.01)
    y1 = mfunc(energies=egrid, pars=pars)

    assert (y1 > 0).any()


@pytest.mark.parametrize('model', MODELS_MUL)
def test_eval_mul(model):
    """Evaluate a multiplicative model."""

    info = x.info(model)

    if model in MODELS_MUL_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    mfunc = getattr(x, model)

    pars = [p.default for p in info.parameters]

    egrid = np.arange(0.1, 10, 0.01)
    y1 = mfunc(energies=egrid, pars=pars)

    assert (y1 > 0).any()


@pytest.mark.parametrize('model', MODELS_CON)
def test_eval_con(model):
    """Evaluate a convolution additive model."""

    info = x.info(model)
    if not info.can_cache:
        pytest.skip(f'Model {model} can not be cached.')

    if model in MODELS_CON_SKIP:
        pytest.skip(f'Model {model} is marked as un-testable.')

    egrid = np.arange(0.1, 10, 0.01)

    def conv(p):
        """what is the default parameter value?"""

        if p.name.casefold() == 'redshift':
            return 0.01

        if p.name.casefold() == 'velocity':
            return 100

        return p.default

    pars = [p.default for p in x.info('powerlaw').parameters]
    mvals = x.powerlaw(energies=egrid, pars=pars)
    mvals = mvals.astype(get_dtype(info))
    ymodel = mvals.copy()

    mfunc = getattr(x, model)

    pars = [conv(p) for p in info.parameters]
    y1 = mfunc(energies=egrid, pars=pars, model=mvals)

    assert (y1 > 0).any()
    assert (y1 != ymodel).any()
