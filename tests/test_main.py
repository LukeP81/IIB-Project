import pytest
import streamlit_mock
import unittest.mock as mock

from main import run
from other.utils import StateEnum


@pytest.mark.parametrize(
    "func, state", [('initialising_func', StateEnum.INITIALISING),
                    ('optimising_func', StateEnum.OPTIMISING),
                    ('interpreting_func', StateEnum.INTERPRETING)]
)
def test_main(func, state):
    """Test that the expected state function is executed"""

    with mock.patch(f'main.{func}') as mock_func:
        sm = streamlit_mock.StreamlitMock()
        session_state = sm.get_session_state()
        session_state['mode'] = state
        run()
        mock_func.assert_called_once()
