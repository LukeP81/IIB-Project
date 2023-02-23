import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backend.plotting import PlotGP
from backend.models import OakModel
import matplotlib.pyplot as plt
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure, show
import tensorflow as tf


class Plot:
    @classmethod
    def contour(cls, model, x_index, y_index):
        x, y, z = PlotGP.plot_2nd_order(model=model, i=x_index, j=y_index)
        fig = go.Figure(data=go.Contour(x=x, y=y, z=z))
        st.plotly_chart(fig)

    @classmethod
    def line(cls, model, dimension):
        (plot_range,
         mean,
         upper,
         lower) = PlotGP.plot_1st_order(model=model,
                                        dimension=dimension)

        mean, upper, lower = (tf.make_ndarray(tf.make_tensor_proto(mean)),
                              tf.make_ndarray(tf.make_tensor_proto(upper)),
                              tf.make_ndarray(tf.make_tensor_proto(lower)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_range, y=mean,
        ))
        fig.add_trace(go.Scatter(
            x=[*plot_range, *plot_range[::-1]],
            y=[*upper, *lower[::-1]],
            fill='toself',
        ))

        st.plotly_chart(fig)
