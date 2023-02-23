import numpy as np
import streamlit as st

from frontend.post_optimisation.plot import Plot
import pandas as pd


class MainPost:
    @classmethod
    def main_post(cls, model):
        with st.expander(label="Hyperparameters"):
            hyperparameters = Hyperparameters(model)
            hyperparameters.display()
        Graphs.first_order(model)
        Graphs.second_order(model)


class Graphs:
    @classmethod
    def first_order(cls, model):
        tabs = st.tabs([f"Dimension {i + 1}"
                        for i in range(model.kernel.num_dims)])
        for dimension, tab in enumerate(tabs):
            with tab:
                Plot.line(model, dimension=dimension)

    @classmethod
    def second_order(cls, model):
        options = [i + 1
                   for i in range(model.kernel.num_dims)]
        x_dimension = st.select_slider("X dimension",
                                       options=options,
                                       format_func=lambda x: f"Dimension {x}")
        y_dimension = st.select_slider("Y dimension",
                                       options=options,
                                       format_func=lambda x: f"Dimension {x}"
                                       )
        if x_dimension == y_dimension:
            st.subheader("Select two different dimensions")
            return
        Plot.contour(model, x_index=x_dimension - 1, y_index=y_dimension - 1)


class Hyperparameters:
    def __init__(self, model):
        self.model = model

    def kernel_parameters(self):
        kernel_parameters = np.array(
            [(kernel.base_kernel.variance.numpy(),
              kernel.base_kernel.lengthscales.numpy())
             for kernel in self.model.kernel.kernels])
        kernel_parameters = {
            "Variance": kernel_parameters[:, 0],
            "Lengthscale": kernel_parameters[:, 1]
        }

        df = pd.DataFrame(kernel_parameters)
        df.index.name = "Base Kernel"
        df.index += 1
        st.dataframe(df)

    def order_parameters(self):
        def order_name(number):
            suffixes = ["th", "st", "nd", "rd"]
            if number % 10 in [1, 2, 3] and number not in [11, 12, 13]:
                return f"{number}{suffixes[number % 10]} Order"
            return f"{number}{suffixes[0]} Order"

        order_parameters = np.array(
            [order.numpy()
             for order in self.model.kernel.order_variance])

        order_parameters = {
            order_name(i): [param]
            for i, param in enumerate(order_parameters)
        }

        df = pd.DataFrame(data=order_parameters, index=["Variance"])
        df.index.name = "Order"
        st.dataframe(df)

    def likelihood(self):
        likelihood = {
            "Likelihood": self.model.likelihood.variance.numpy()
        }

        df = pd.DataFrame(data=likelihood, index=["Variance"])
        st.dataframe(df)

    def display(self):
        self.kernel_parameters()
        self.order_parameters()
        self.likelihood()
