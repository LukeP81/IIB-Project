import gpflow
import mat4py
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import tensorflow as tf
from backend.models import OakModel


class FileDetails:
    @classmethod
    def details(cls, file):
        if isinstance(file, str):
            file_data = cls.example_file(file)
        else:
            file_data = cls.uploaded_file(file)

        file_dataframe = pd.DataFrame(file_data)
        file_dataframe.index += 1
        st.dataframe(file_dataframe)

        keys = file_data.keys()
        x_label, y_label = st.columns(2)
        with x_label:
            x_key = st.selectbox("X data label", options=keys)
        with y_label:
            y_key = st.selectbox("Y data label", options=keys, index=1)

        run = st.button("press")

        if run:
            x_data, y_data = file_data[x_key], file_data[y_key]
            CacheModel.cache_model(x_data, y_data)

    @classmethod
    def example_file(cls, filename: str) -> dict:
        st.subheader(f"Filename: Example-{filename}.mat")
        return mat4py.loadmat(filename=f"data/{filename}.mat")

    @classmethod
    def uploaded_file(cls, file: UploadedFile):
        st.subheader(f"Filename: {file.name}")
        return mat4py.loadmat(file)


class CacheModel:
    @classmethod
    def cache_model(cls, x, y):
        def printing(*args):
            chart.add_rows(np.reshape([num.numpy() for num in args[2]], (1, -1)))

        model = OakModel.create(x, y)
        vals = np.reshape([val.numpy() for val in
                           gpflow.utilities.parameter_dict(model).values()],
                          (1, -1))
        df = pd.DataFrame(vals)

        chart = st.line_chart(df)

        model = OakModel.optimize(model, callback=printing)
        model_params = gpflow.utilities.parameter_dict(model)
        st.session_state["model_attributes"] = {"dims": model.kernel.num_dims,
                                                "data": model.data,
                                                "params": model_params}
        st.experimental_rerun()
