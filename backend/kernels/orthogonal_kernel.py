"""Module for implementing an orthogonal squared exponential kernel"""

from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf


# pylint-disable=abstract-method
class OrthogonalSEKernel(gpflow.kernels.Kernel):
    """
    Class for implementing the constrained squared exponential kernel
    :param active_dims: active dimension
    :return: constrained SE kernel
    """

    def __init__(self, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.base_kernel = gpflow.kernels.SquaredExponential()
        self.active_dims = self.active_dims
        self.measure_var = 1

        def covariance(x):
            tf.debugging.assert_shapes([(x, (..., "N", 1))])
            length = self.base_kernel.lengthscales
            sigma2 = self.base_kernel.variance
            mean, var = 0, 1
            return (
                    sigma2
                    * length
                    / tf.sqrt(length ** 2 + var)
                    * tf.exp(-0.5 * ((x - mean) ** 2) / (length ** 2 + var))
            )

        def variance():
            length = self.base_kernel.lengthscales
            sigma2 = self.base_kernel.variance
            return sigma2 * length / tf.sqrt(length ** 2 + 2 * 1)

        self.covariance = covariance
        self.variance = variance

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        :param X: input array X
        :param X2: input array X2, if None, set to X
        :return: kernel matrix K(X,X2)
        """
        covariance_x = self.covariance(X)
        covariance_x2 = covariance_x if X2 is None else self.covariance(X2)

        to_subtract = tf.tensordot(
            covariance_x,
            tf.transpose(covariance_x2),
            axes=1
        ) / self.variance()

        return self.base_kernel(X, X2) - to_subtract

    def K_diag(self, X):
        covariance = self.covariance(X)
        k = self.base_kernel.K_diag(X) - tf.square(covariance[:, 0]
                                                   ) / self.variance()
        return k
