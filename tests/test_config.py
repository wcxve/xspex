from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from xspec import Xset

import xspex as xx

from .conftest import XSPEC_ABUND_TABLES, XSPEC_XSECT_TABLES


def test_xspec_version():
    """Test XSPEC version retrieval.

    Verifies that xspex reports the same XSPEC version as PyXspec.
    This ensures the underlying XSPEC library version is correctly detected.
    """
    assert xx.xspec_version() == Xset.version[1]


def test_chatter():
    """Test XSPEC chatter levels.

    Tests the ability to set different XSPEC console chatter levels
    and verify that the setting is correctly applied. Tests common
    chatter levels: 0 (silent), 5 (quiet), 10 (normal), 25 (verbose).

    The test ensures that the chatter level is properly restored
    to its original value after testing.
    """
    # Save original chatter level
    original_chatter = xx.chatter()

    try:
        # Test setting different chatter levels
        for level in [0, 5, 10, 25]:
            xx.chatter(level)
            assert xx.chatter() == level
    finally:
        # Restore original chatter level
        xx.chatter(original_chatter)


def test_abund():
    """Test abundance tables and files functionality.

    Tests the ability to set different abundance tables using both
    predefined table names and custom abundance files. Verifies that:

        1. Default abundance table matches XSPEC after initialization
        2. All standard XSPEC abundance tables can be set and retrieved
        3. Custom abundance files can be loaded from disk
        4. Invalid table names raise appropriate errors
        5. Invalid file paths raise appropriate errors
        6. Original settings are properly restored

    The test covers common abundance tables including angr, aspl, feld,
    aneb, grsa, wilm, lodd, lpgp, and lpgs.
    """
    # Save original abundance table
    original_abund = xx.abund()

    # Test default abundance table matches XSPEC after initialization
    assert xx.abund() == Xset.abund[:4]

    try:
        # Test setting different abundance tables
        for table in XSPEC_ABUND_TABLES:
            xx.abund(table)
            current_abund = xx.abund()
            assert current_abund.startswith(table)

        # Test setting abundance file
        abund_list = Xset.abund.split(':')[1].strip()
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(abund_list.replace(' ', '\n').encode())
            tmpfile.flush()
            abund_file = Path(tmpfile.name)
            abund_file = abund_file.resolve(strict=True)
            xx.abund_file(abund_file.as_posix())
            current_abund = xx.abund()
            assert current_abund == 'file:' + abund_file.as_posix()
    finally:
        # Restore original abundance table
        xx.abund(original_abund)

    # Test setting abundance table with invalid name
    with pytest.raises(ValueError):
        xx.abund('????')

    # Test setting abundance file with invalid path
    with pytest.raises(ValueError):
        xx.abund_file('./abund.txt')

    # Verify the original abundance table is unchanged
    assert xx.abund() == original_abund


def test_xsect():
    """Test photo-electric cross-section tables functionality.

    Tests the ability to set different photo-electric cross-section tables
    and verify that the settings are correctly applied. Verifies that:

        1. Default cross-section table matches XSPEC after initialization
        2. All standard XSPEC cross-section tables can be set and retrieved
        3. Invalid table names raise appropriate errors
        4. Original settings are properly restored

    The test covers the main cross-section tables: bcmc, obcm, and vern.
    """
    # Save original xsect table
    original_xsect = xx.xsect()

    # Test default cross-section table matches XSPEC after initialization
    assert original_xsect == Xset.xsect

    try:
        # Test setting different xsect tables
        for table in XSPEC_XSECT_TABLES:
            xx.xsect(table)
            assert xx.xsect() == table
    finally:
        # Restore original xsect table
        xx.xsect(original_xsect)

    # Test setting xsect table with invalid name
    with pytest.raises(ValueError):
        xx.xsect('????')

    # Verify the original xsect table is unchanged
    assert xx.xsect() == original_xsect


def test_cosmo():
    """Test cosmological parameters functionality.

    Tests the ability to set and retrieve cosmological parameters used
    in XSPEC models. Verifies that:

        1. Default cosmological parameters match XSPEC after initialization
        2. H0 (Hubble constant in km/s/Mpc) can be set and retrieved accurately
        3. q0 (deceleration parameter) can be set and retrieved accurately
        4. lambda0 (cosmological constant) can be set and retrieved accurately
        5. Multiple parameter combinations work correctly
        6. Original settings are properly restored

    Tests both standard cosmological models (ΛCDM with different parameters)
    and verifies numerical precision in floating-point comparisons.
    """
    # Save original cosmological parameters
    original_cosmo = xx.cosmo()

    # Test default cosmological parameters match XSPEC after initialization
    xs_cosmo = Xset.cosmo
    for i, key in enumerate(('H0', 'q0', 'lambda0')):
        assert_allclose(
            original_cosmo[key],
            float(xs_cosmo[i]),
            err_msg=key,
        )

    try:
        # Test setting different cosmological parameters
        test_params = [
            {
                'H0': 70.0,
                'q0': 0.0,
                'lambda0': 0.73,
            },
            {
                'H0': 67.8,
                'q0': -0.55,
                'lambda0': 0.692,
            },
        ]

        for params in test_params:
            xx.cosmo(params['H0'], params['q0'], params['lambda0'])
            current_cosmo = xx.cosmo()

            # Use approximate comparison for float values
            assert_allclose(
                current_cosmo['H0'],
                params['H0'],
                err_msg='H0 mismatch',
            )
            assert_allclose(
                current_cosmo['q0'],
                params['q0'],
                err_msg='q0 mismatch',
            )
            assert_allclose(
                current_cosmo['lambda0'],
                params['lambda0'],
                err_msg='lambda0 mismatch',
            )
    finally:
        # Restore original cosmological parameters
        original_h0 = original_cosmo['H0']
        original_q0 = original_cosmo['q0']
        original_lambda0 = original_cosmo['lambda0']
        xx.cosmo(original_h0, original_q0, original_lambda0)

    # Verify the original cosmological parameters are unchanged
    cosmo = xx.cosmo()
    assert_allclose(cosmo['H0'], original_cosmo['H0'])
    assert_allclose(cosmo['q0'], original_cosmo['q0'])
    assert_allclose(cosmo['lambda0'], original_cosmo['lambda0'])


def test_mstr():
    """Test model strings functionality.

    Comprehensive test of model string (MSTR) functionality including:

        1. The database is empty initially
        2. Setting single model strings with key-value pairs
        3. Setting multiple model strings using dictionaries
        4. Retrieving individual model strings by key
        5. Retrieving all model strings as a dictionary
        6. Overwriting existing model strings
        7. The clear operation removes all model strings
        8. The database is empty after clearing

    """
    # Get all model strings (should return empty dict initially)
    all_mstr = xx.mstr()
    assert all_mstr == {}

    try:
        # Clear all model strings first
        xx.clear_mstr()

        # Test setting single model string
        test_key = 'test_key'
        test_value = 'test_value'
        xx.mstr(test_key, test_value)

        # Verify the model string was set
        retrieved_value = xx.mstr(test_key)
        assert retrieved_value == test_value

        # Test setting multiple model strings
        test_dict = {
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }
        xx.mstr(test_dict)

        # Verify all model strings were set
        all_mstr = xx.mstr()
        for key, expected_value in test_dict.items():
            assert key in all_mstr
            assert all_mstr[key] == expected_value
            assert xx.mstr(key) == expected_value

        # Test overwriting existing model string
        xx.mstr('key1', 'new_value1')
        assert xx.mstr('key1') == 'new_value1'

    finally:
        # Clear all model strings
        xx.clear_mstr()

        # Database should be empty after clearing
        assert xx.mstr() == {}


def test_xflt():
    """Test XFLT functionality.

    Comprehensive test of XFLT functionality including:

        1. Retrieving XFLT entries (empty database and specific spectra)
        2. Setting XFLT entries for individual and multiple spectra
        3. Verifying XFLT entry retrieval and data integrity
        4. Selective clearing of XFLT entries for specific spectra
        5. Global clearing of all XFLT entries

    """
    try:
        # Test 1: Initial state is empty
        xx.clear_xflt()
        assert xx.xflt() == {}

        # Test getting XFLT for a specific spectrum number (empty)
        assert xx.xflt(1) == {}

        # Test 2: Set XFLT for a specific spectrum (1)
        test_xflt_single = {'test_key': 1.0}
        xx.xflt(1, test_xflt_single)

        # Verify the XFLT was set
        retrieved_xflt = xx.xflt(1)
        for key, expected_value in test_xflt_single.items():
            assert key in retrieved_xflt
            assert retrieved_xflt[key] == expected_value

        # Test 3: Set XFLT for multiple spectra (2 and 3)
        test_xflt_multi = {
            2: {'test_key': 2.0},
            3: {'test_key': 3.0},
        }
        xx.xflt(test_xflt_multi)

        # Verify all XFLT entries were set
        all_xflt = xx.xflt()
        for spec_num_multi, expected_dict in test_xflt_multi.items():
            assert spec_num_multi in all_xflt
            for key, expected_value in expected_dict.items():
                assert all_xflt[spec_num_multi][key] == expected_value

        # Test 4: Clear XFLT for specific spectrum
        xx.clear_xflt(1)
        all_xflt_after_clear = xx.xflt()
        assert 1 not in all_xflt_after_clear
        assert 2 in all_xflt_after_clear  # Should still exist
        assert 3 in all_xflt_after_clear  # Should still exist

    finally:
        # Clear all XFLT entries
        xx.clear_xflt()
        assert xx.xflt() == {}


def test_list_models():
    """Test model listing functionality.

    Tests the ability to list available XSPEC models by type and verify
    the completeness and correctness of the model registry. Verifies that:

        1. All models can be listed without type filter
        2. Additive models can be listed separately
        3. Multiplicative models can be listed separately
        4. Convolution models can be listed separately
        5. The sum of typed models equals all models
        6. No duplicate models exist in the listings
        7. Unsupported model types raise NotImplementedError
        8. Invalid model types raise ValueError

    This ensures the model registry is properly structured and accessible.
    """
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
    """Test model function and information retrieval.

    Tests the model retrieval system which provides access to XSPEC model
    functions and their metadata. Verifies that:

        1. Model functions and info objects are singletons (same instance
        returned for repeated calls)
        2. All listed models can be successfully retrieved
        3. Model functions and info objects are properly paired
        4. Invalid model names raise ValueError
        5. Unsupported model types raise NotImplementedError

    The singleton pattern ensures efficient memory usage and consistent
    model instances across the application. Model info contains metadata
    like parameter definitions, energy ranges, and model descriptions.
    """
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
