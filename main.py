"""Main file"""

import streamlit as st

from other.utilities import AppStates, clear_cache
from initialising.initialise_home import Initialise
from optimising.optimise_home import Optimize
from interpreting.interpret_home import Interpret


def page_setup():
    """Configure the page"""

    st.set_page_config(page_title="OAK GP Interpretation",
                       page_icon="🌳",
                       layout="wide")
    # st.title("OAK GP Interpretation")
    # todo sidebar


def run() -> None:
    """Launches the application in the correct state"""

    page_setup()

    current_state = st.session_state.get('mode', None)

    if current_state is None:
        clear_cache()
        st.session_state['mode'] = AppStates.INITIALISING
        current_state = AppStates.INITIALISING

    state_func = {
        AppStates.INITIALISING: Initialise.display,
        AppStates.OPTIMISING: Optimize.display,
        AppStates.INTERPRETING: Interpret.display,
    }
    state_func[current_state]()


if __name__ == "__main__":
    run()
