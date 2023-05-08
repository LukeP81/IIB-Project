"""Module for accessing essential features of the OAK model"""
import tensorflow as tf
from oak.model_utils import oak_model


def generate_oak_model(training_x: tf.Tensor,
                       training_y: tf.Tensor,
                       model_kw: dict,
                       fit_kw: dict
                       ) -> oak_model:
    pass
