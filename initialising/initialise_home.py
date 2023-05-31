"""Module for the initialisation stage of the application"""
from typing import Any, Optional

import mat4py
import streamlit as st

from cacheable_api.api_exceptions import NotCachedError
from cacheable_api.file_api import FileAPI
from initialising.file_selection import FileSelector
from initialising.file_details import UploadedDetails, load_example, \
    FileFormatError
from other.utilities import AppStates
from optimising.optimise_home import Optimize


def extract_uploaded_data(file: Any) -> dict:
    """Potential placeholder for allowing different file types"""
    return mat4py.loadmat(filename=file)


def extract_example_data(filename: str) -> dict:
    """Potential placeholder for allowing different file types"""
    return mat4py.loadmat(filename=f"./data/{filename}.mat")


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
        widget_frame = st.empty()
        if isinstance(file, str):
            file_data = extract_example_data(filename=file)
            details_dict = load_example(file_data=file_data, filename=file)
        else:

            file_data = extract_uploaded_data(file=file)
            try:

                with widget_frame.container():
                    details_dict = UploadedDetails.display(file_data=file_data,
                                                           filename=file.name)
            except FileFormatError:
                st.write("message")
                # todo: come up with suitable message and example
                return

        if details_dict is not None:
            widget_frame.empty()
            FileAPI.set_file_data(file_x_data=details_dict["x"],
                                  file_y_data=details_dict["y"])
            FileAPI.set_feature_names(data_feature_names=details_dict["f"],
                                      data_value_name=details_dict["v"])
            st.session_state["mode"] = AppStates.OPTIMISING
            Optimize.display()

    @classmethod
    def display(cls) -> None:
        """Method displaying initialisation UI"""
        try:
            saved_file = FileAPI.get_saved_file()
        except NotCachedError:
            saved_file = None

        if saved_file is not None:
            cls._details(saved_file)

        else:
            st.cache_data.clear()

            widget_frame = st.empty()
            with widget_frame:
                file = cls._select()

            if file is not None:
                widget_frame.empty()

                FileAPI.set_saved_file(file=file)

                cls._details(file)
