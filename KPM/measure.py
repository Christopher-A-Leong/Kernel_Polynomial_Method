import numpy as np
import scipy
import scipy.sparse as sparse


def rescale_operator_unity_window(operator, eigval_min, eigval_max, epsilon=0.01):
    a = (eigval_max - eigval_min) / (2 - epsilon)
    b = (eigval_max + eigval_min) / 2
    return (operator - (sparse.identity(operator.shape[0], format='csr') * b)) / a


def rescale_disorder_unity_window(disorder_vals, eigval_min, eigval_max, epsilon=0.01):
    a = (eigval_max - eigval_min) / (2 - epsilon)
    return disorder_vals / a


def chebyshev_recursion_measurement_calc(spectrum_vals, prevprev_term, prev_term):
    return 2 * spectrum_vals * prev_term - prevprev_term


def rescale_spectrum_unity_window(spectrum_vals, eigval_min, eigval_max, epsilon=0.01):
    a = (eigval_max - eigval_min) / (2 - epsilon)
    b = (eigval_max + eigval_min) / 2

    return (spectrum_vals - b) / a


def jackson_kernel(moments):
    N = len(moments)
    n = np.arange(len(moments))
    numer1 = (N - n + 1) * np.cos((np.pi * n) / (N + 1))
    numer2 = np.sin((np.pi * n) / (N + 1)) / np.tan(np.pi / (N + 1))
    return (numer1 + numer2) / (N + 1)


def calculate_ADOS_from_moments(moments, spectrum_vals,
                                eigval_min, eigval_max, epsilon=0.01,
                                kernel_func=jackson_kernel):

    rescaled_spectrum_vals = rescale_spectrum_unity_window(spectrum_vals, eigval_min, eigval_max, epsilon=epsilon)

    summation_result = np.zeros(len(rescaled_spectrum_vals), dtype=np.float64)
    prevprev_sum_term = None
    prev_sum_term = None

    moments = moments * kernel_func(moments)
    if len(moments) >= 0:
        prevprev_sum_term = np.repeat(1, len(spectrum_vals))
        summation_result = summation_result + moments[0]

    if len(moments) >= 1:
        prev_sum_term = rescaled_spectrum_vals
        summation_result = summation_result + 2 * moments[1] * rescaled_spectrum_vals

    if len(moments) >= 2:
        for i in range(2, len(moments)):
            new_sum_term = chebyshev_recursion_measurement_calc(rescaled_spectrum_vals,
                                                                prevprev_sum_term,
                                                                prev_sum_term)
            prevprev_sum_term = prev_sum_term
            prev_sum_term = new_sum_term
            summation_result = summation_result + 2 * moments[i] * prev_sum_term

    return (summation_result /
            (np.pi * np.sqrt(1 - rescaled_spectrum_vals**2)) /
            ((eigval_max - eigval_min) / (2 - epsilon)))


def calculate_LDOS_from_moments(moments, spectrum_vals,
                                eigval_min, eigval_max, epsilon=0.01,
                                kernel_func=jackson_kernel):

    rescaled_spectrum_vals = rescale_spectrum_unity_window(spectrum_vals, eigval_min, eigval_max, epsilon=epsilon)

    summation_result = np.zeros((moments.shape[1], len(rescaled_spectrum_vals)), dtype=np.float64)
    prevprev_sum_term = None
    prev_sum_term = None

    moments = moments * kernel_func(moments).reshape(-1, 1)
    if len(moments) >= 0:
        prevprev_sum_term = np.full((moments.shape[1], len(rescaled_spectrum_vals)), 1.0)
        summation_result = summation_result + moments[0, :].reshape(-1, 1)

    if len(moments) >= 1:
        prev_sum_term = np.tile(rescaled_spectrum_vals, moments.shape[1]).reshape(moments.shape[1], -1)
        summation_result = summation_result + 2 * moments[1, :].reshape(-1, 1) * rescaled_spectrum_vals

    if len(moments) >= 2:
        for i in range(2, len(moments)):
            new_sum_term = chebyshev_recursion_measurement_calc(rescaled_spectrum_vals,
                                                                prevprev_sum_term,
                                                                prev_sum_term)
            prevprev_sum_term = prev_sum_term
            prev_sum_term = new_sum_term
            summation_result = summation_result + 2 * moments[i, :].reshape(-1, 1) * prev_sum_term

    return (summation_result /
            (np.pi * np.sqrt(1 - rescaled_spectrum_vals**2)) /
            ((eigval_max - eigval_min) / (2 - epsilon)))


