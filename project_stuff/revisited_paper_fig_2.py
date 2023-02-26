import gpflow
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from backend.kernels.additive_kernel import OAK
from backend.kernels.orthogonal_kernel import OrthogonalSEKernel
from backend.predictions import PlotGP


def x_data():
    n_points = 31
    x_range = np.linspace(-1, 1, n_points)
    x1, x2 = np.meshgrid(x_range, x_range)
    x_points = np.stack([x1, x2], axis=-1)
    return x_points.reshape([n_points ** 2, 2])


def get_y(x):
    x1, x2 = x

    return (np.square(x1) - 2 * x2 +
            np.cos(3 * x1) * np.sin(5 * x2)) + np.random.normal(loc=0, scale=0.01)


def generate_data():
    x = x_data()
    y = np.apply_along_axis(get_y, 1, x)
    return x, np.reshape(y, (-1, 1))


def cw1e():
    x, y = generate_data()
    X = tf.cast(x, tf.float64)
    Y = tf.cast(y, tf.float64)

    model = gpflow.models.GPR(
        (X, Y), kernel=OAK(num_dims=2, base_kernel=OrthogonalSEKernel)
    )
    opt = gpflow.optimizers.Scipy()

    opt.minimize(model.training_loss, model.trainable_variables,
                 options={"disp": True})
    gpflow.utilities.print_summary(model)
    plot_range = np.linspace(-1.5, 1.5, 301)
    PlotGP.predict_1st_order(model, 0, plot_range)
    PlotGP.predict_1st_order(model, 1, plot_range)
    plot_range = np.linspace(-50, 50, 301)
    # x,y,z=PlotGP.plot_2nd_order(model, 0, 1, plot_range, plot_range)
    # plt.contour(x,y,z)
    # plt.show()
    x,y=PlotGP.predict_2nd_order(model, 0, 1, plot_range, plot_range, marginal=0)
    plt.plot(x,y)
    plt.show()
    x,y=PlotGP.predict_2nd_order(model, 0, 1, plot_range, plot_range, marginal=1)
    plt.plot(x,y)
    plt.show()



cw1e()

# todo remaining plots
# print(x_data())
# marginal is when other dim is 0, therfore cos is 1 so sin can work but sin 0 so no cos at all
