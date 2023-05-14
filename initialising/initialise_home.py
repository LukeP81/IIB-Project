"""Module for the initialisation stage of the application"""
from typing import Any, Optional

import mat4py
import streamlit as st

from cacheable_api.file_api import get_saved_file
from initialising.file_selection import FileSelector
from initialising.file_details import UploadedDetails, load_example, \
    FileFormatError


def extract_data(file: Any) -> dict:
    """Potential placeholder for allowing different file types"""
    return mat4py.loadmat(filename=file)


class Initialise:  # pylint: disable = too-few-public-methods
    """Class for holding main display methods for initialisation"""

    @classmethod
    def _select(cls) -> Optional[Any]:
        """Method for file selection

        :return: the selected file
        """
        with st.container():
            return FileSelector.display()

    @classmethod
    def _details(cls, file: Any) -> None:
        """Method for setting file details
        :param file: the file being used
        """

        if isinstance(file, str):
            load_example(filename=file)
        else:
            with st.container():
                file_data = extract_data(file=file)
                try:
                    UploadedDetails.display(file_data=file_data,
                                            filename=file.name)
                except FileFormatError:
                    st.write("message")
                    # todo: come up with suitable message and example

    @classmethod
    def display(cls) -> None:
        """Method displaying initialisation UI"""

        saved_file = get_saved_file()

        if saved_file is not None:
            cls._details(saved_file)

        else:
            st.cache_data.clear()

            placeholder = st.empty()
            with placeholder:
                file = cls._select()

            if file is not None:
                placeholder.empty()

                st.session_state["saved_file"] = file
                _ = get_saved_file()

                cls._details(file)
