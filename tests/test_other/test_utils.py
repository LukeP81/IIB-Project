"""Test utils.py"""
from other.utils import StateEnum


def test_enum_options():
    """Check for the correct order and amount of values"""
    assert list(StateEnum) == [StateEnum.INITIALISING,
                               StateEnum.OPTIMISING,
                               StateEnum.INTERPRETING]


def test_enum_values():
    """Check that all values are set correctly"""
    assert StateEnum.INITIALISING.value == "initialising"
    assert StateEnum.OPTIMISING.value == "optimising"
    assert StateEnum.INTERPRETING.value == "interpreting"
