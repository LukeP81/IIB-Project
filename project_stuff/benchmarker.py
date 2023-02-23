from time import perf_counter
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def benchmark(func1: callable,
              func2: callable,
              func_args: List[tuple],
              repetitions: Optional[int] = 10,
              average_over: Optional[int] = 5,
              assert_equality: Optional[bool] = True,
              equality_func: Optional[callable] = None,
              filename: Optional[str] = None,
              func1_name: Optional[str] = None,
              func2_name: Optional[str] = None,
              arg_names: Optional[List[str]] = None,
              log_time: Optional[bool] = False
              ) -> None:
    def check_equality() -> None:
        if assert_equality:
            for args in func_args:
                func1_result = func1(*args)
                func2_result = func2(*args)
                if equality_func is not None:
                    equality_func(func1_result, func2_result)
                else:
                    assert func1_result == func2_result

        else:  # run both so tensorflow/other setup doesn't affect timings
            _ = func1(*func_args[0])
            _ = func2(*func_args[0])

    def get_timings() -> Tuple[np.ndarray, np.ndarray]:
        def timing(function, args):
            start = perf_counter()
            for _ in range(repetitions):
                function(*args)
            end = perf_counter()
            return end - start

        func_timings = np.array([[timing(func1, arg_set), timing(func2, arg_set)]
                                 for arg_set in func_args
                                 for _ in range(average_over)])
        if log_time:
            func_timings = np.log(func_timings)
        data_shape = (-1, average_over)

        return (np.reshape(func_timings[:, 0], data_shape),
                np.reshape(func_timings[:, 1], data_shape))

    def plotting() -> None:
        arg_nums = len(func_args)

        x = [f"Arguments #{i}" if arg_names is None else arg_names[i]
             for i in range(arg_nums)]

        func1_means = [np.mean(data) for data in func1_data]
        func2_means = [np.mean(data) for data in func2_data]
        func1_95 = [2 * np.std(data) for data in func1_data]
        func2_95 = [2 * np.std(data) for data in func2_data]

        func1_label = "Function 1" if func1_name is None else func1_name
        func2_label = "Function 2" if func2_name is None else func2_name

        # plt.plot(x, func1_means, label=func1_label)
        plt.errorbar(x, func1_means, func1_95, capsize=3, label=func1_label)
        # plt.plot(x, func2_means, label=func2_label)
        plt.errorbar(x, func2_means, func2_95, capsize=3, label=func2_label)

        plt.xlabel("Arguments")
        plt.ylabel("Time (s)" if not log_time else "Log Time (s)")
        plt.legend()
        min_time = np.min([func1_data, func2_data])
        max_time = np.max([func1_data, func2_data])
        plt.ylim(0 if not log_time else min_time * 1.1, max_time * 1.1)
        plt.tight_layout()

    check_equality()
    func1_data, func2_data = get_timings()
    plotting()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
