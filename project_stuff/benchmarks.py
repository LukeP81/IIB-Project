from functools import reduce

import numpy as np
import tensorflow as tf

from benchmarker import benchmark


def newton_girard_s_term():
    def power_by_pow(kernel_matrices, num_dims):
        s = [
            reduce(tf.add, [tf.pow(k, p) for k in kernel_matrices])
            for p in range(1, num_dims + 1)
        ]
        return s

    def power_by_multiply(kernel_matrices, num_dims):
        orders = [kernel_matrices]
        for p in range(1, num_dims):
            orders.append([tf.multiply(orders[-1][k], orders[0][k])
                           for k in range(num_dims)])
        return [reduce(tf.add, order) for order in orders]

    args = [([np.random.randn(size, size) for _ in range(num_dims)], num_dims)
            for num_dims in [5, 10, 15] for size in [250, 500, 1000]]
    name_list = [f"D = {num_dims}\nS = {size}"
                 for num_dims in [5, 10, 15] for size in [250, 500, 1000]]
    benchmark(func1=power_by_pow,
              func2=power_by_multiply,
              func_args=args,
              equality_func=tf.experimental.numpy.allclose,
              filename="Newton-Girard_S_term",
              func1_name="Powers by .pow",
              func2_name="Powers by consecutive .multiply",
              arg_names=name_list)


newton_girard_s_term()
