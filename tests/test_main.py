"""Test main.py"""

from unittest.mock import patch

import pytest
import streamlit

from main import run
from other.utilities import AppStates


def test_startup():
    """Test that the expected startup routine is executed"""

    with (
        patch('streamlit.session_state', new={}) as session_state,
        patch('main.Initialise.display') as mock_func,
        patch('main.clear_cache') as mock_clear_cache
    ):
        run()

        assert "mode" in session_state
        assert session_state["mode"] == AppStates.INITIALISING
        mock_func.assert_called_once()
        mock_clear_cache.assert_called_once()


@patch('streamlit.session_state', new={})
@pytest.mark.parametrize(
    "func, state", [('Initialise.display', AppStates.INITIALISING),
                    ('Optimize.display', AppStates.OPTIMISING),
                    ('Interpret.display', AppStates.INTERPRETING)]
)
def test_main(func, state):
    """Test that the expected state function is executed"""

    with patch(f'main.{func}') as mock_func:
        streamlit.session_state['mode'] = state
        run()

        mock_func.assert_called_once()
