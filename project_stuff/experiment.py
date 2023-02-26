import gpflow.kernels
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

import gpflow

from backend.kernels.additive_kernel import OAK
from backend.kernels.orthogonal_kernel import OrthogonalSEKernel

def cw1a():
    matfile = loadmat(file_name="data/cw1a")
    X, Y = matfile["x"], matfile["y"]
    plt.plot(X, Y, "kx", mew=2)
    plt.show()
    model = gpflow.models.GPR(
        (X, Y),
        kernel=OAK(num_dims=1)
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    Xnew = np.array([[0.5]])
    model.predict_f(Xnew)
    model.predict_y(Xnew)
    Xplot = np.linspace(-3, 3, 601)[:, None]
    y_mean, y_var = model.predict_y(Xplot)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)
    plt.plot(X, Y, "kx", mew=2, label="input data")
    plt.plot(Xplot, y_mean  )
    plt.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    )
    plt.legend()
    plt.show()
    gpflow.utilities.print_summary(model)


cw1a()

def basic(step, variables, values):
    print(step)
    print(variables)
    print(values)


def cw1e():
    matfile = loadmat(file_name="data/servo")
    X, Y = matfile["X"], matfile["y"]
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    # plt.plot(X, Y, "kx", mew=2)
    # plt.show()
    n_grid = 601
    # hide: begin
    # fmt: off
    # hide: end
    # X = np.array(
    #     [
    #         [-0.4, -0.5], [0.1, -0.3], [0.4, -0.4], [0.5, -0.5], [-0.5, 0.3],
    #         [0.0, 0.5], [0.4, 0.4], [0.5, 0.3],
    #     ]
    # )
    # Y = np.array([[0.8], [0.0], [0.5], [0.3], [1.0], [0.2], [0.7], [0.5]])
    # hide: begin
    # fmt: on
    # hide: end
    model = gpflow.models.GPR(
        (X, Y), kernel=OAK(num_dims=4)
    )
    opt = gpflow.optimizers.Scipy()

    opt.minimize(model.training_loss, model.trainable_variables,
                 options={"disp": True})

    # Xplots = np.linspace(-10, 10, n_grid)
    # Xplot1, Xplot2 = np.meshgrid(Xplots, Xplots)
    # Xplot = np.stack([Xplot1, Xplot2], axis=-1)
    # Xplot = Xplot.reshape([n_grid ** 2, 2])
    #
    # f_mean, _ = model.predict_f(Xplot, full_cov=False)
    # f_mean = f_mean.numpy().reshape((n_grid, n_grid))
    # fig,ax =plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(Xplot1, Xplot2, f_mean, cmap=coolwarm, alpha=0.7)
    # ax.scatter(X[:, 0], X[:, 1], Y[:, 0], s=50, c="black")
    # ax.set_title("Example data fit")
    gpflow.utilities.print_summary(model)
    # plt.show()


cw1e()
# num_dims = 2
# kernel_matrices = [np.ones((2, 2)) * 2 for i in range(num_dims)]

# def current_func():
#   return [
#       [tf.pow(k, p) for k in kernel_matrices]
#       for p in range(1, num_dims + 1)
#    ]
# current_func()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
