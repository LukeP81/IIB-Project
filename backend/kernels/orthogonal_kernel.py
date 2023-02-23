import gpflow
import numpy as np
import tensorflow as tf
from typing import Optional


class OrthogonalRBFKernel(gpflow.kernels.Kernel):
    """
    :param active_dims: active dimension
    :return: constrained BRF kernel
    """

    def __init__(
            self, active_dims=None
    ):
        super().__init__(active_dims=active_dims)
        self.base_kernel = gpflow.kernels.SquaredExponential()
        self.active_dims = self.active_dims
        self.measure_var = 1

        def cov_X_s(X):
            tf.debugging.assert_shapes([(X, (..., "N", 1))])
            length = self.base_kernel.lengthscales
            sigma2 = self.base_kernel.variance
            mu, var = 0, 1
            return (
                    sigma2
                    * length
                    / tf.sqrt(length ** 2 + var)
                    * tf.exp(-0.5 * ((X - mu) ** 2) / (length ** 2 + var))
            )

        def var_s():
            length = self.base_kernel.lengthscales
            sigma2 = self.base_kernel.variance
            return sigma2 * length / tf.sqrt(length ** 2 + 2 * 1)

        self.cov_X_s = cov_X_s
        self.var_s = var_s

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        :param X: input array X
        :param X2: input array X2, if None, set to X
        :return: kernel matrix K(X,X2)
        """
        cov_X_s = self.cov_X_s(X)
        if X2 is None:
            cov_X2_s = cov_X_s
        else:
            cov_X2_s = self.cov_X_s(X2)
        k = (
                self.base_kernel(X, X2)
                - tf.tensordot(cov_X_s, tf.transpose(cov_X2_s), 1) / self.var_s()
        )
        return k

    def K_diag(self, X):
        cov_X_s = self.cov_X_s(X)
        k = self.base_kernel.K_diag(X) - tf.square(cov_X_s[:, 0]) / self.var_s()
        return k
