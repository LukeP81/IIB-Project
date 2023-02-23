from functools import reduce

import tensorflow as tf


class NewtonGirard:

    @classmethod
    def _s_term(cls, num_dims, kernel_matrices):
        s = [
            reduce(tf.add, [tf.pow(k, p) for k in kernel_matrices])
            for p in range(1, num_dims + 1)
        ]
        return s

    @classmethod
    def _e_term(cls, num_dims, kernel_matrices, s):
        e = [tf.ones_like(kernel_matrices[0])]  # start with constant term
        for n in range(1, num_dims + 1):
            e.append(
                (1.0 / n)
                * reduce(
                    tf.add,
                    [((-1) ** (k - 1)) * e[n - k] * s[k - 1] for k in
                     range(1, n + 1)],
                )
            )
        return e

    @classmethod
    def compute(cls, num_dims, kernel_matrices):
        s = cls._s_term(num_dims, kernel_matrices)
        return cls._e_term(num_dims, kernel_matrices, s)
