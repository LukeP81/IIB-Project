"""Module for implementing the Newton-Girard method for computing combinations"""

from functools import reduce
from typing import List

import gpflow.kernels
import tensorflow as tf


class NewtonGirard:  # pylint:disable=too-few-public-methods
    """Namespace for methods implementing the Newton-Girard formulae"""

    @classmethod
    def _s_terms(cls,
                 num_dims: int,
                 base_kernels: List[gpflow.kernels.Kernel]
                 ) -> List[gpflow.kernels.Kernel]:
        """
        Private method for computing the s terms of the Newton-Girard formulae
        :param num_dims: the number of dimensions of the input data
        :param base_kernels: the base kernels to use
        :return: the s terms of the Newton-Girard formulae
        """

        orders = [base_kernels]
        for _ in range(1, num_dims):
            orders.append([tf.multiply(orders[-1][k], orders[0][k])
                           for k in range(num_dims)])
        return [reduce(tf.add, order) for order in orders]

    @classmethod
    def _e_terms(cls,
                 num_dims: int,
                 base_kernels: List[gpflow.kernels.Kernel],
                 s_terms: List[gpflow.kernels.Kernel]
                 ) -> List[gpflow.kernels.Kernel]:
        """
        Private method for computing the e terms of the Newton-Girard formulae
        :param num_dims: the number of dimensions of the input data
        :param base_kernels: the base kernels to use
        :param s_terms: the s terms of the Newton-Girard formulae
        :return: the e terms of the Newton-Girard formulae
        """

        e_term = [tf.ones_like(base_kernels[0])]  # start with constant term
        for order in range(1, num_dims + 1):
            e_term.append(
                (1.0 / order)
                * reduce(tf.add,
                         [((-1) ** (k - 1)) * e_term[order - k] * s_terms[k - 1]
                          for k in range(1, order + 1)], )
            )
        return e_term

    @classmethod
    def compute(cls,
                num_dims: int,
                base_kernels: List[gpflow.kernels.Kernel],
                ) -> List[gpflow.kernels.Kernel]:
        """
        Method for computing combinations using the Newton-Girard formulae
        :param num_dims: the number of dimensions of the input data
        :param base_kernels: the base kernels to use
        :return: the sum of combinations of base kernels for each order
        """

        s_terms = cls._s_terms(num_dims=num_dims,
                               base_kernels=base_kernels)
        return cls._e_terms(num_dims=num_dims,
                            base_kernels=base_kernels,
                            s_terms=s_terms)
