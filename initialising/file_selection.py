"""Module for file methods"""

from typing import Any, Optional

import streamlit as st
import pandas as pd


class FileSelector:
    """Namespace for file methods"""

    @classmethod
    def load_model(cls):
        """todo: probably place in extra tab"""

    @classmethod
    def display(cls) -> Optional[Any]:
        """Method for displaying UI for file selection
        :return: None, str, or UploadedFile. The UploadedFile class caused testing
        issues when importing directly so has been replaced with Any.
        """

        st.subheader("File Selection")
        upload_tab, example_tab = st.tabs(["Upload File", "Example File"])

        with upload_tab:
            file = cls._upload()
            if file is not None:
                return file

        with example_tab:
            file = cls._examples()
            if file is not None:
                return file

        return None

    @classmethod
    def _select_file(cls, options, key_name):
        """Private method for selecting from a list of files"""
        file_selected = st.selectbox(label="Selected File",
                                     options=options,
                                     index=0,
                                     key=f"initial_file_select_{key_name}",
                                     help="Use the dropdown box to select a file")

        if file_selected is None:
            return None

        if st.button(label="Use File",
                     key=f"using_selected_file_{key_name}",
                     help="Click this to use the selected file"):
            return file_selected

        return None

    @classmethod
    def _upload(cls) -> Optional[Any]:
        """Private method for file uploader"""
        uploaded_files = st.file_uploader(label="Upload files",
                                          type="mat",
                                          accept_multiple_files=True,
                                          key="initial_file_upload",
                                          help="Upload files to be used")

        if not uploaded_files or uploaded_files is None:
            uploaded_names = [None]
        elif not isinstance(uploaded_files, list):
            uploaded_names = [uploaded_files.name]
            uploaded_files = [uploaded_files]
        else:
            uploaded_names = [file.name for file in uploaded_files]

        file_name = cls._select_file(options=uploaded_names,
                                     key_name="upload")
        if file_name is None:
            return None
        return uploaded_files[uploaded_names.index(file_name)]

    @classmethod
    def _examples(cls) -> Optional[str]:
        """Private method for example files"""

        file_index = ["Dimensions", "Datapoints"]
        files_data = {"cw1a": ["1", "75"],
                      "cw1e": ["2", "121"],
                      "concrete": ["8", "1030"],
                      "pima": ["8", "768"],
                      "servo": ["4", "167"]}

        example_file_df = pd.DataFrame(data=files_data,
                                       index=file_index)
        example_file_df.index.name = "Filename"

        st.write("Example Files")
        st.dataframe(data=example_file_df)

        example_files = list(files_data.keys())

        file_name = cls._select_file(options=example_files,
                                     key_name="example")
        return file_name
