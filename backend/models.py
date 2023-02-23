import gpflow
import tensorflow as tf
from backend.kernels.additive_kernel import AdditiveKernel
from backend.kernels.orthogonal_kernel import OrthogonalRBFKernel


class OakModel:
    @classmethod
    def create(cls, x_data, y_data):
        x_data = tf.cast(x_data, tf.float64)
        y_data = tf.cast(y_data, tf.float64)
        num_dims = x_data.shape[1]
        return gpflow.models.GPR(
            (x_data, y_data),
            kernel=AdditiveKernel(num_dims=num_dims,
                                  base_kernel=OrthogonalRBFKernel)
        )

    @classmethod
    def optimize(cls, model, callback):
        opt = gpflow.optimizers.Scipy()

        opt.minimize(model.training_loss, model.trainable_variables,
                     step_callback=callback)
        return model

    @classmethod
    def pre_run(cls, dims, data, params):
        model = gpflow.models.GPR(
            data=data,
            kernel=AdditiveKernel(num_dims=dims, base_kernel=OrthogonalRBFKernel)
        )
        gpflow.utilities.multiple_assign(model, params)
        return model
