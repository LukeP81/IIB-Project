"""Main file"""

import streamlit as st
from other.utilities import AppStates, clear_cache


def initialising_func():
    """placeholder"""


def optimising_func():
    """placeholder"""


def interpreting_func():
    """placeholder"""


def run() -> None:
    """Launches the application in the correct state"""

    current_state = st.session_state.get('mode', None)

    if current_state is None:
        clear_cache()
        st.session_state['mode'] = AppStates.INITIALISING
        current_state = AppStates.INITIALISING

    state_func = {
        AppStates.INITIALISING: initialising_func,
        AppStates.OPTIMISING: optimising_func,
        AppStates.INTERPRETING: interpreting_func,
    }
    state_func[current_state]()


if __name__ == "__main__":
    run()
