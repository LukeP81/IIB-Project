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
