from other.utils import StateEnum


def test_values():
    assert StateEnum.INITIALISING.value == "initialising"
    assert StateEnum.OPTIMISING.value == "optimising"
    assert StateEnum.INTERPRETING.value == "interpreting"


def test_state_iteration():
    assert list(StateEnum) == [StateEnum.INITIALISING,
                               StateEnum.OPTIMISING,
                               StateEnum.INTERPRETING]
