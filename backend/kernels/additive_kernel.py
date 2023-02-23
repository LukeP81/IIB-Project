import gpflow
from functools import reduce
import tensorflow as tf

from backend.kernels.additive_terms import NewtonGirard
from backend.kernels.orthogonal_kernel import OrthogonalRBFKernel


class AdditiveKernel(gpflow.kernels.Kernel):

    def __init__(
            self,
            num_dims: int,
            base_kernel=OrthogonalRBFKernel
    ):
        super().__init__(active_dims=range(num_dims))

        self.kernels = [
            base_kernel(active_dims=[dim])
            for dim in range(num_dims)
        ]
        self.num_dims = num_dims
        self.order_variance = [
            gpflow.Parameter(1, transform=gpflow.utilities.positive())
            for _ in range(num_dims + 1)
        ]

    def K(self, X, X2=None):
        kernel_matrices = [
            k(X, X2) for k in self.kernels
        ]  # note that active dims gets applied by each kernel
        additive_terms = NewtonGirard.compute(self.num_dims, kernel_matrices)
        return reduce(
            tf.add,
            [sigma2 * k for sigma2, k in
             zip(self.order_variance, additive_terms)],
        )

    def K_diag(self, X):
        kernel_diags = [k.K_diag(k.slice(X, None)[0]) for k in self.kernels]
        additive_terms = NewtonGirard.compute(self.num_dims, kernel_diags)

        return reduce(
            tf.add,
            [sigma2 * k for sigma2, k in
             zip(self.order_variance, additive_terms)],
        )
