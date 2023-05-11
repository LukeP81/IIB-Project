"""Module for the initialisation stage of the application"""
import streamlit as st

from initialising.file_select import FileSelector


def details_display():
    """placeholder"""


class Initialise:
    """Class for holding main display methods for initialisation"""

    @classmethod
    def display(cls):
        """Main display for initialisation"""

        holder = st.empty()

        with holder:
            file = cls._select()

        if file is not None:
            with holder:
                cls._details()

    @classmethod
    def _select(cls):
        with st.container():
            return FileSelector.display()

    @classmethod
    def _details(cls):
        details_display()
