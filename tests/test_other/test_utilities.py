"""Test utilities.py"""

from unittest.mock import patch

from other.utilities import AppStates, clear_cache


def test_enum_options():
    """Check for the attributes"""

    class_attr = dir(AppStates)

    assert "INITIALISING" in class_attr
    assert "OPTIMISING" in class_attr
    assert "INTERPRETING" in class_attr


def test_enum_values():
    """Check that all values are set correctly"""

    assert AppStates.INITIALISING == "initialising"
    assert AppStates.OPTIMISING == "optimising"
    assert AppStates.INTERPRETING == "interpreting"


def test_clearing():
    """Check whether the cache is cleared properly"""

    with (patch('streamlit.cache_data.clear') as data_clear,
          patch('streamlit.cache_resource.clear') as resource_clear):
        clear_cache()

        data_clear.assert_called_once()
        resource_clear.assert_called_once()
