"""Module for file methods"""
from typing import List, Union

import streamlit as st
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile


class FileUtilities:
    """Namespace for file methods"""

    @classmethod
    def file_selector(cls) -> Union[None, str, UploadedFile]:
        """Method for displaying UI for file selection"""

        def on_file_change():
            """Removes the saved model parameters"""
            st.session_state.pop("model_attributes", None)

        uploaded_files = cls._upload()
        uploaded_names = [file.name for file in uploaded_files]
        if not uploaded_names:
            uploaded_names = [None]

        example_files = cls._examples()
        example_names = [f"Example - {file}" for file in example_files]

        file_selected = st.sidebar.selectbox(label="Selected File",
                                             options=[*uploaded_names,
                                                      *example_names],
                                             on_change=on_file_change)
        if file_selected is None:
            return None

        if file_selected in uploaded_names:
            return uploaded_files[uploaded_names.index(file_selected)]

        return example_files[example_names.index(file_selected)]

    @classmethod
    def _upload(cls) -> List[UploadedFile]:
        """Private method for file uploader"""

        return st.sidebar.file_uploader(label="Upload files",
                                        type="mat",
                                        accept_multiple_files=True)

    @classmethod
    def _examples(cls) -> List[str]:
        """Private method for example files"""

        example_files = ["cw1a", "cw1e"]
        example_file_df = pd.DataFrame(data={
            "Dimensions": ["1", "2"],
            "Datapoints": ["75", "121"], }, index=example_files)
        example_file_df.index.name = "Filename"
        st.sidebar.write("Example Files")
        st.sidebar.dataframe(data=example_file_df)
        return example_files

    @classmethod
    def no_file_selected(cls) -> None:
        """Method for displaying UI when no file is selected"""

        st.header(
            "Please upload a file using the sidebar or select an example file")
