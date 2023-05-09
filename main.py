"""Main file"""

import streamlit as st
from other.utils import StateEnum


def initialising_func():
    """placeholder"""
    pass


def optimising_func():
    """placeholder"""
    pass


def interpreting_func():
    """placeholder"""
    pass


current_state = st.session_state.get('mode', None)
if current_state is None:
    st.session_state['mode'] = StateEnum.INITIALISING

state_func = {
    StateEnum.INITIALISING: initialising_func,
    StateEnum.OPTIMISING: optimising_func,
    StateEnum.INTERPRETING: interpreting_func,
}

state_func[current_state]()
