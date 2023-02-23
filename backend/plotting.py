import numpy as np

import tensorflow as tf


class PlotGP:
    @classmethod
    def plot_1st_order(cls, model, dimension, xx=None, granularity=100):
        x_data, y_data = model.data
        x_min, x_max = np.min(x_data[:, dimension]), np.max(x_data[:, dimension])
        base_kernel = model.kernel.kernels[dimension]
        order_variance = model.kernel.order_variance[1]
        plot_range = np.linspace(x_min, x_max, granularity) if xx is None else xx

        kxx = (
                base_kernel.K(plot_range[:, None],
                              x_data[:, dimension:dimension + 1])
                * model.kernel.order_variance[1]
        )
        k = model.kernel(model.data[0])
        k_tilde = k + np.eye(model.data[0].shape[0]) * model.likelihood.variance
        chol = np.linalg.cholesky(k_tilde)
        alpha = tf.linalg.cholesky_solve(chol, model.data[1])
        mean = tf.matmul(kxx, alpha)[:, 0]

        tmp = tf.linalg.triangular_solve(chol, tf.transpose(kxx))
        var = base_kernel.K_diag(plot_range[:, None]) * order_variance - np.sum(
            tmp ** 2, axis=0)

        lower = mean - 2 * np.sqrt(var)
        upper = mean + 2 * np.sqrt(var)
        return plot_range, mean, upper, lower

    @classmethod
    def plot_2nd_order(cls, model, i, j, xx=None, yy=None, granularity=100,
                       marginal=None):
        x_data, y_data = model.data

        Xi = x_data[:, i].numpy()
        Xj = x_data[:, j].numpy()

        X_conditioned = x_data

        xmin, xmax = Xi.min(), Xi.max()
        ymin, ymax = Xj.min(), Xj.max()

        k = model.kernel(model.data[0])
        ktilde = k + np.eye(model.data[0].shape[0]) * model.likelihood.variance
        l = np.linalg.cholesky(ktilde)
        alpha = tf.linalg.cholesky_solve(l, model.data[1])
        xx_range = np.linspace(start=xmin, stop=xmax,
                               num=101) if xx is None else xx
        yy_range = np.linspace(start=ymin, stop=ymax,
                               num=101) if yy is None else yy

        xx, yy = np.meshgrid(xx_range, yy_range)
        XX = np.vstack([xx.flatten(), yy.flatten()]).T
        Kxx = (
                model.kernel.kernels[i].K(XX[:, 0:1], X_conditioned[:, i: i + 1])
                * model.kernel.order_variance[2]
        )
        Kxx *= model.kernel.kernels[j].K(XX[:, 1:2], X_conditioned[:, j: j + 1])
        mu = np.dot(Kxx, alpha)

        if marginal is not None:
            mu_reshaped = mu.reshape(*xx.shape)
            marginal_vals = np.mean(mu_reshaped, axis=marginal)
            return (xx_range, marginal_vals)

        else:
            return (xx_range, yy_range, mu.reshape(*xx.shape))
