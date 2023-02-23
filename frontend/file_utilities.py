import streamlit as st
import pandas as pd


class FileUtilities:
    @classmethod
    def file_selector(cls):
        def on_file_change():
            st.session_state.pop("model_attributes", None)

        uploaded_files = cls.upload()
        uploaded_names = [file.name for file in uploaded_files]
        if not uploaded_names:
            uploaded_names = [None]

        example_files = cls.examples()
        example_names = [f"Example - {file}" for file in example_files]

        file_selected = st.sidebar.selectbox(label="Selected File",
                                             options=[*uploaded_names,
                                                      *example_names],
                                             on_change=on_file_change)
        if file_selected is None:
            return

        if file_selected in uploaded_names:
            return uploaded_files[uploaded_names.index(file_selected)]

        return example_files[example_names.index(file_selected)]

    @classmethod
    def upload(cls):
        return st.sidebar.file_uploader(label="Upload files",
                                        type="mat",
                                        accept_multiple_files=True)

    @classmethod
    def examples(cls):
        example_files = ["cw1a", "cw1e"]
        df = pd.DataFrame(data={
            "Dimensions": ["1", "2"],
            "Datapoints": ["75", "121"], }, index=example_files)
        df.index.name = "Filename"
        st.sidebar.write("Example Files")
        st.sidebar.dataframe(data=df)
        return example_files

    @classmethod
    def not_selected(cls):
        st.header(
            "Please upload a file using the sidebar or select an example file")
