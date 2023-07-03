"""Module for interpretation"""
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from cacheable_api.file_api import FileAPI
from cacheable_api.oak_api import ModelAPI
from cacheable_api.computation_api import ComputationAPI

from cacheable_api.plot_api import PlotAPI
import plotly.graph_objects as go


class Interpret:
    """Class holding methods for displaying interpretations"""

    @staticmethod
    def display():
        """Main display method"""
        oak = ModelAPI.get_oak_model()
        with st.spinner("Computing model details"):
            computed_data = ComputationAPI.get_data(oak)

        cumulative_sobol = computed_data["cumulative_sobol"]
        nll = computed_data["nll"]
        normalised_sobols = computed_data["normalised_sobols"]
        order = computed_data["order"]
        r2 = computed_data["r2"]
        rmse = computed_data["rmse"]
        rmse_component = computed_data["rmse_component"]
        tuple_of_indices = computed_data["tuple_of_indices"]

        st.title("Model Summary")
        metric_col, orders_col, components_col = st.columns([2, 5, 5])
        with metric_col:
            st.header("Metrics")
            Interpret.show_dataset_metrics()
            Interpret.show_performance_metrics(nll, r2, rmse)

        with orders_col:
            st.header("Order Contributions")
            Interpret.plot_order_sobol(normalised_sobols, tuple_of_indices)

        with components_col:
            st.header("Component Contributions")
            Interpret.plot_rmse_sobol(cumulative_sobol, order,
                                      rmse_component)

        amount = st.number_input(label="Components to plot", min_value=1,
                                 max_value=len(order), value=min(5, len(order)))

        with st.spinner("Generating component plots"):
            covariate_names = FileAPI.get_covariate_names()
        fig_list = Interpret.fig_listing(amount, covariate_names, oak)

        with st.expander(label="Component Plots", expanded=False):
            Interpret.plot_components(fig_list)

    @staticmethod
    def plot_components(fig_list):
        """Plots component graphs"""
        left_col, middle_col, right_col = st.columns(3)
        col_dict = {0: left_col,
                    1: middle_col,
                    2: right_col}
        for i, fig in enumerate(fig_list):
            if i % 3 == 0:
                left_col, middle_col, right_col = st.columns(3)
                col_dict = {0: left_col,
                            1: middle_col,
                            2: right_col}
            with col_dict[i % 3]:
                if fig is None:
                    st.write("Component of higher order than 2")
                else:
                    st.pyplot(fig.fig)

    @staticmethod
    def plot_rmse_sobol(cumulative_sobol, order, rmse_component):
        """Plots RMSE vs Sobol graph"""
        x_vals = np.arange(len(order))
        x_vals += 1
        plt.figure(figsize=(8, 4))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(x_vals, rmse_component, "r", linewidth=4)
        ax2.plot(x_vals, cumulative_sobol, "-.g", linewidth=4)
        ax1.set_xlabel("Number of Terms Added")
        ax1.set_ylabel("RMSE", color="r")
        ax2.set_ylabel("Cumulative Sobol", color="g")
        plt.tight_layout()
        # st.pyplot(plt)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x_vals, y=rmse_component,
                                 mode='lines',
                                 line={"color": "red", "width": 4},
                                 name='RMSE'), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_vals, y=cumulative_sobol,
                                 mode='lines',
                                 line={"color": 'green', "width": 4,
                                       "dash": 'dash'},
                                 name='Cumulative Sobol'), secondary_y=True)
        fig.update_layout(
            title='RMSE and Cumulative Sobol',
            xaxis_title='Number of Terms Added',
            yaxis=dict(
                title='RMSE',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'), showgrid=False
                # Remove horizontal gridlines
            ),
            yaxis2=dict(
                title='Cumulative Sobol',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                side='right', showgrid=False  # Remove horizontal gridlines
            ),
            showlegend=True,

        )
        st.plotly_chart(fig)

    @staticmethod
    def plot_order_sobol(normalised_sobols, tuple_of_indices):
        """Plots Sobol order histogram"""
        # aggregate sobol per order of interactions
        sobol_order = np.zeros(len(tuple_of_indices[-1]))
        for i in range(len(tuple_of_indices)):
            sobol_order[len(tuple_of_indices[i]) - 1] += normalised_sobols[i]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=np.arange(1, len(sobol_order) + 1), y=sobol_order))
        fig.update_layout(
            title_text='Sum of Sobol Indices of Orders',
            xaxis_title='Order',
            yaxis_title='Sobol Indices',
            xaxis=dict(
                tickmode='linear',
                dtick=1
            ),
            bargap=0.2,
            bargroupgap=0.1
        )
        st.plotly_chart(fig)

    @staticmethod
    def fig_listing(amount, covariate_names, oak):
        """Gets the figure list"""
        fig_list = PlotAPI.get_component_figures(
            amount=amount,
            _covariate_names=covariate_names,
            _oak=oak)
        return fig_list

    @staticmethod
    def show_performance_metrics(nll, r2, rmse):
        """Shows performance metrics"""
        st.subheader("Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Test RMSE",
                      value=np.round(rmse, 4))
            st.metric(label="Test R\u00b2",
                      value=np.round(r2, 4))
        with col2:
            st.metric(label="Test NLL",
                      value=np.round(nll, 4))

    @staticmethod
    def show_dataset_metrics():
        """Shows dataset metrics"""
        x_data, y_data = FileAPI.get_file_data()
        x_test, y_test = ModelAPI.get_test_data()
        data_shape = np.shape(x_data)
        test_shape = np.shape(x_test)
        st.markdown("### Dataset")
        col1, col2 = st.columns(2)

        col1.metric(label="Total datapoints",
                    value=data_shape[0])
        col2.metric(label="Dimensions",
                    value=data_shape[1])

        col1.metric(label="Training datapoints",
                    value=data_shape[0] - test_shape[0])
        col2.metric(label="Test datapoints",
                    value=test_shape[0])
