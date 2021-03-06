import pandas as pd
import numpy as np
import message
import strings
import lbfgs
import math

dic = strings.UTILITIES_EN


# input parameter
# @feature should be type(np.array)
# @type should be type(str)
def normalize(dataset, features, norm_type):
    dataset = np.array(dataset, dtype='float64')
    features = list(dataset.columns) if features == [] else features

    if norm_type == "std":
        return normalize_std(dataset, features)
    elif norm_type == "rescaling":
        return normalize_rescaling(dataset, features)


def normalize_std(dataset, features):
    dataset_norm = dataset.astype('float64')
    features = list(dataset.columns) if features == [] else features

    for feature in features:
        feature_mean = dataset_norm[feature].mean()
        feature_std = 1 if dataset_norm[feature].std() == 0. else dataset_norm[feature].std()
        dataset_norm.loc[:, feature] = dataset_norm[feature].apply(lambda x: (x - feature_mean) / feature_std)

    return dataset_norm


def normalize_rescaling(dataset, features):
    dataset_norm = dataset.astype('float64')
    features = list(dataset.columns) if features == [] else features

    for feature in features:
        feature_min = dataset_norm[feature].min()
        feature_max = dataset_norm[feature].max()
        length = 1. if (feature_max - feature_min) == 0. else feature_max - feature_min

        dataset_norm.loc[:, feature] = dataset_norm[feature].apply(lambda x: (x - feature_min) / float(length))

    return dataset_norm


def polynomial_features(dataset, features, degree):
    # dataset = pd.DataFrame(dataset, dtype='float64')
    poly_data = pd.DataFrame(dtype='float64')

    for feature in features:
        data = dataset[feature]
        for i in range(1, degree + 1):
            feature_name = feature + "_" + str(i)
            poly_data[feature_name] = data.apply(lambda x: x ** i)
    # poly_data = data_frame_rescaling(poly_data)
    return poly_data


def polynomial_feature(feature, degree):
    dim = feature.ravel().shape[0]
    feature = pd.Series(feature.ravel())
    poly_data = pd.DataFrame()
    for i in range(1, degree + 1):
        feature_name = "power_" + str(i)
        poly_data[feature_name] = feature.apply(lambda x: x ** i)
    return poly_data


# waiting to optimize
def data_frame_rescaling(data_frame):
    # data_frame = pd.DataFrame(data_frame, dtype='O')
    # feature_names = list(data_frame.columns)
    # print(data_frame[feature_names[-1]])
    # MAX = 2 ** 32


    # for feature in feature_names:
    #     max_frame = data_frame[feature].max()
    #     denominator = max_frame / float(MAX)
    # if max_frame <= MAX:
    #     continue
    # data_frame[feature].apply(lambda x: x / denominator)

    return data_frame


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_feature = feature_matrix / norms
    return normalized_feature, norms


def euclidean_distance(target, dataset, weights=None):
    target = np.array(target, dtype='float64')
    dataset = np.array(dataset, dtype='float64')
    if weights is None:
        weights = np.ones(1 if len(dataset.shape) == 1 else dataset.shape[1])
    weights = np.array(weights, dtype='float64').ravel()

    weights_matrix = np.diag(weights)
    errors = target - dataset

    weighted_error = np.dot(errors, weights_matrix)
    sum_of_square = (weighted_error * errors).sum(axis=1)
    distances = np.sqrt(sum_of_square).ravel()
    return distances


def gaussian_kernel(x, bandwidth):
    x = np.array(x, dtype='float64')
    weights = np.exp(-(x * x) / bandwidth)
    return weights if len(x.shape) == 1 else weights.sum(axis=1)


def sigmoid(scores):
    scores = np.array(scores, dtype='float64')
    f = 1 / (1 + np.exp(-scores))
    return f


def sigmoid_gradient(scores):
    scores = np.array(scores, dtype='float64')
    f = sigmoid(scores)
    g = (1 - f) * f
    return g


def relu(scores):
    scores = np.array(scores)
    f = np.array(list(
        map(lambda x: x if x > 0 else 0, scores)
    ))
    return f


def relu_gradient(scores):
    scores = np.array(scores)
    g = np.array(list(
        map(lambda x: 1 if x > 0 else 0, scores)
    ))
    return g


def softplus(scores):
    scores = np.array(scores)
    f = np.log(1 + np.exp(scores))
    return f


def softplus_gradient(scores):
    scores = np.array(scores)
    g = 1. - 1 / (1 + np.exp(scores))
    return g


def one_hot_encoder(vector,num_of_class):
    vector = np.array(vector)
    code = np.zeros((len(vector), num_of_class))
    index = vector
    for i in range(len(vector)):
        code[i][index[i]] = 1.

    return code


# cost_function@param receive at least 4 parameter: feature_matrix,weights,output
# calculate_derivative@param receive at least 4 parameter: feature_matrix,weights,output
def l_bfgs(feature_matrix,
           output,
           initial_weights,
           cost_function,
           calculate_derivative,
           iter_times=500,
           l2_penalty=0.,
           silent_mode=False,
           ):
    reporter = message.Reporter(silent_mode)

    feature_matrix = np.array(feature_matrix, dtype='float64')
    weights = np.array(initial_weights, dtype='float64')
    output = np.array(output, dtype='float64')

    reporter.report("Starting L-BFGS")
    costs = []

    # index 0 : the optimal x
    # index 1 : the iteration times k
    infos = [0, 0]

    optimal_weights = [0]

    def f(x, g):
        gradients = calculate_derivative(feature_matrix, x, output, l2_penalty)
        g[:len(gradients)] = gradients
        return cost_function(feature_matrix, x, output, l2_penalty)

    def progress(x, g, cost, xnorm, gnorm, step, k, ls):
        """
            x	The current values of variables.
            g	The current gradient values of variables.
            fx	The current value of the objective function.
            xnorm	The Euclidean norm of the variables.
            gnorm	The Euclidean norm of the gradients.
            step	The line-search step used for this iteration.
            n	The number of variables.
            k	The iteration count.
            ls	The number of evaluations called for this iteration.
        """
        reporter.report("%10s:\t%d\t\t%10s:%f" % (dic['ITER'], k, dic['COST'], cost))
        costs.append(cost)
        infos[0] = x
        infos[1] = k
        optimal_weights[0] = x

        return 0 if k < iter_times else 1

    try:
        lbfgs.fmin_lbfgs(f, weights, progress)
    except Exception:
        reporter.report("Max Iteration reach.")

    reporter.report()

    optimal_weights = np.array(infos[0])
    iter_times = infos[1]


    return optimal_weights, costs, iter_times



