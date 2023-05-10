"""Test main.py"""

from unittest.mock import patch

import pytest
import streamlit_mock

from main import run
from other.utilities import AppStates


def test_startup():
    """Test that the expected startup routine is executed"""

    with (
        patch.dict('streamlit.session_state', {}) as session_state,
        patch('main.initialising_func') as mock_func,
        patch('main.clear_cache') as mock_clear_cache
    ):
        run()

        assert "mode" in session_state
        assert session_state["mode"] == AppStates.INITIALISING
        mock_func.assert_called_once()
        mock_clear_cache.assert_called_once()


@pytest.mark.parametrize(
    "func, state", [('initialising_func', AppStates.INITIALISING),
                    ('optimising_func', AppStates.OPTIMISING),
                    ('interpreting_func', AppStates.INTERPRETING)]
)
def test_main(func, state):
    """Test that the expected state function is executed"""

    with patch(f'main.{func}') as mock_func:
        st_mock = streamlit_mock.StreamlitMock()
        session_state = st_mock.get_session_state()
        session_state['mode'] = state
        run()

        mock_func.assert_called_once()
