import numpy as np
import strings
import message
from math import sqrt

dic = strings.UTILITIES_EN


# input parameter
# return parameter
# scalar cost(float)
# error n-dim vector(np.array)
def lr_cost_function(feature_matrix: np.ndarray, weights: np.ndarray, output: np.ndarray, l2_penalty: float) -> float:
    """

    :param feature_matrix:
    :param weights:
    :param output:
    :param l2_penalty:
    :return:
    """
    feature_matrix = np.array(feature_matrix)
    output = np.array(output).ravel()
    weights = np.array(weights).ravel()

    m = feature_matrix.shape[0]

    predictions = np.dot(feature_matrix, weights)
    errors = np.array(predictions - output)

    cost = ((errors * errors).sum() + l2_penalty * (np.dot(weights, weights))) / (2 * m)

    return cost


# return
# n-dim vector (np.array)
def lr_derivative(feature_matrix, weights, output, l2_penalty) -> np.ndarray:
    output = output.ravel()
    predictions = np.dot(feature_matrix, weights)
    errors = np.array(predictions - output)

    m = feature_matrix.shape[0]
    errors = np.array(errors)
    feature_matrix = np.array(feature_matrix)

    # derivatives = 2 * np.dot(errors, feature_matrix)
    derivatives = np.dot(errors, feature_matrix) / m
    derivatives = derivatives.ravel()

    # do not regularize the constant term
    derivatives = np.append(derivatives[:-1] + l2_penalty * weights[:-1] / m, derivatives[-1])

    return derivatives.ravel()


def gradient_descent(feature_matrix,
                     output,
                     initial_weights,
                     cost_function,
                     calculate_derivative,
                     step_size,
                     iter_times=500,
                     tolerance=0.,
                     l2_penalty=0.,
                     silent_mode=False,
                     ):
    reporter = message.Reporter(silent_mode)

    feature_matrix = np.array(feature_matrix, dtype='float64')
    weights = np.array(initial_weights, dtype='float64')
    output = np.array(output, dtype='float64')

    reporter.report(dic['STAT'])
    costs = []
    for i in range(int(iter_times)):
        cost = cost_function(feature_matrix, weights, output, l2_penalty)
        derivatives = calculate_derivative(feature_matrix, weights, output, l2_penalty)

        weights -= step_size * derivatives

        costs.append(cost)
        reporter.report("%10s:\t%d\t\t%10s:%f" % (
            dic['ITER'], i, dic['COST'], cost))

        gradient_sum_squares = np.dot(derivatives, derivatives)
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            break

    return np.array(weights), costs


def lr_least_square(feature_matrix, output, l2_penalty):
    Y = np.mat(output)
    X = np.mat(feature_matrix)
    ID = l2_penalty * np.mat(np.identity(X.shape[1]))
    ID[X.shape[1] - 1, X.shape[1] - 1] = 1.

    weights = ((X.T * X) + ID).I * X.T * Y

    weights = np.array(weights).ravel()
    cost = lr_cost_function(feature_matrix, weights, output, l2_penalty)
    return weights, cost


def coordinate_descent_step(j, feature_matrix, weights, output, l1_penalty=0., l2_penalty=0.):
    feature_matrix = np.array(feature_matrix)
    output = np.array(output).ravel()
    weights = np.array(weights).ravel()

    # feature and weight without j-th value
    feature_minus_j = np.append(feature_matrix[:, :j], feature_matrix[:, j + 1:], axis=1)
    weights_minus_j = np.append(weights[:j], weights[j + 1:])

    predictions = np.dot(feature_minus_j, weights_minus_j)
    errors = output - predictions

    rho = np.dot(errors, feature_matrix[:, j])
    z = np.dot(feature_matrix[:, j], feature_matrix[:, j])

    # do not optimize constant term
    if j == (len(weights) - 1):
        update_weight = rho
    # when w_j < 0
    elif rho < -l1_penalty / 2.:
        update_weight = (rho + l1_penalty / 2) / (z + l2_penalty)
    # when w_j > 0
    elif rho > l1_penalty / 2.:
        update_weight = (rho - l1_penalty / 2) / (z + l2_penalty)
    # when w_j = 0
    else:
        update_weight = 0.

    return update_weight


def coordinate_descent(feature_matrix,
                       weights,
                       output,
                       iter_times=500,
                       tolerance=0.,
                       l1_penalty=0.,
                       l2_penalty=0.,
                       silent_mode=False):
    feature_matrix = np.array(feature_matrix)
    output = np.array(output).ravel()
    weights = np.array(weights).ravel()

    reporter = message.Reporter(silent_mode)

    costs =[]

    reporter.report(dic["STAT_CD"])
    for i in range(iter_times):
        max_step_size = 0
        for j in range(len(weights)):
            old_weight = weights[j]
            weights[j] = coordinate_descent_step(j, feature_matrix, weights, output, l1_penalty=l1_penalty,
                                                 l2_penalty=l2_penalty)
            change = abs(old_weight - weights[j])

            if change > max_step_size:
                max_step_size = change

        cost = lr_cost_function(feature_matrix, weights, output, l2_penalty)
        costs.append(cost)
        reporter.report("%10s:\t%d\t\t%10s:%f" % (
            dic['ITER'], i, dic['COST'], cost))

        if max_step_size < tolerance:
            break

    return weights, costs
