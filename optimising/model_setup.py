"""Module for setting up the model"""

import numpy as np
import streamlit as st

from cacheable_api.file_api import FileAPI


class ModelSetup:
    """Class containing methods for setting up the model"""

    @staticmethod
    def optimise_display():
        """Main display method for model setup"""

        st.title("Model Setup")

        x_data, y_data = FileAPI.get_file_data()

        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        num_dims = np.shape(x_data)[1]
        num_points = np.shape(x_data)[0]

        data_col, model_col = st.columns(2)
        with data_col:
            with st.expander("Data options", expanded=True):
                x_data, y_data = ModelSetup.shuffling(num_points=num_points,
                                                      x_data=x_data,
                                                      y_data=y_data)

                split_data = ModelSetup.splitting(num_points=num_points,
                                                  x_data=x_data,
                                                  y_data=y_data)
                x_train, x_test, y_train, y_test = split_data
        with model_col:
            model_kwargs = ModelSetup.options(num_dims=num_dims,
                                              num_points=num_points)

        _, next_col, _ = st.columns([5, 1, 5])

        with next_col:
            if st.button(label="Optimise Model",
                         key="model_select_optimise_button",
                         help="Press this to optimise the model"):
                return {"x_train": x_train,
                        "y_train": y_train,
                        "x_test": x_test,
                        "y_test": y_test,
                        "model_kwargs": model_kwargs
                        }
        return None

    @staticmethod
    def splitting(num_points, x_data, y_data):
        """Creates test/train split"""
        data_split = st.slider(label="Train/test split", min_value=50.0,
                               max_value=95.0, value=80.0, step=0.1,
                               format="%.1f%%")
        split = int(np.round(num_points * data_split / 100))
        x_train, x_test = x_data[:split], x_data[split:]
        y_train, y_test = y_data[:split], y_data[split:]
        num_train = len(x_train)
        num_test = len(x_test)
        train_col, test_col = st.columns(2)
        with train_col:
            st.metric(label="Training points", value=num_train)
        with test_col:
            st.metric(label="Testing points", value=num_test)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def options(num_dims, num_points):
        """Shows model options"""
        with st.expander(label="Model options", expanded=True):
            max_interaction_depth = st.number_input(
                label="Maximum interaction depth",
                min_value=1, max_value=num_dims,
                value=2,
                step=1,
                key="model_options_interaction_depth",
                help="maximum number of interaction terms to consider"
            )
            if num_points > 1000:
                num_inducing = st.slider(
                    label="Inducing points",
                    min_value=100, max_value=1000,
                    value=200,
                    step=1,
                    key="model_options_num_inducing",
                    help="Number of inducing points for sparse GP"
                )
            else:
                num_inducing = 200
            min_lengthscale = st.number_input(
                label="Minimum lengthscale",
                value=1e-3,
                format="%e",
                key="model_options_min_lengthscale",
                help="maximum number of interaction terms to consider"
            )
            max_lengthscale = st.number_input(
                label="Maximum lengthscale",
                value=1e3,
                format="%e",
                key="model_options_max_lengthscale",
                help="maximum number of interaction terms to consider"
            )
            lengthscale_bounds = [min_lengthscale, max_lengthscale]

            model_kwargs = {"max_interaction_depth": max_interaction_depth,
                            "num_inducing": num_inducing,
                            "lengthscale_bounds": lengthscale_bounds,
                            }
        return model_kwargs

    @staticmethod
    def shuffling(num_points, x_data, y_data):
        """Randomizes data"""
        shuffle_data = st.checkbox(
            label="Shuffle data", value=True,
            key="data_options_shuffle_data",
            help="Whether to randomize the order of the datapoints (recommended)")
        if shuffle_data:
            idx = np.random.permutation(range(num_points))
            x_data = x_data[idx, :]
            y_data = y_data[idx]
        return x_data, y_data
