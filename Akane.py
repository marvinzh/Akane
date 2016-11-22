import pandas as pd
import numpy as np
import models
import regression
import classification
import time
import message
import utilities
import strings
import clustering
from akane_exception import *

dic = strings.AKANE_EN


def linear_regression(dataset: pd.DataFrame,
                      target: list(str),
                      features: list(str),
                      initial_weights: list = None,
                      l1_penalty: float = 0.,
                      l2_penalty: float = 0.,
                      step_size: float = 0.,
                      max_iteration: int = 0,
                      solver='auto',
                      tolerance: float = 0.,
                      silent_mode: bool = False):
    """
    Function to create a Simple Linear Regression model, by adding L1 or L2 penalty term, to use the Lasso/Ridge
    Regression, Elastic Net. Note that if you add L1 penalty(set L1_penalty non-zero) to model, akane will be forced
    to train model by Coordinate Descent Algorithm.

    :param dataset: should be a Pandas DataFrame, with each data point stored in row-wise.
    :param target: should be a list Object(python built-in)
    :param features: should be a list Object(python built-in)
    :param initial_weights: should be a list Object(python built-in)
    :param l1_penalty: should be a float Object(python built-in)
    :param l2_penalty: should be a float Object(python built-in)
    :param step_size: should be a float Object(python built-in)
    :param max_iteration: should be a integer Object(python built-in), note that the model may be converged before
    maximum iteration
    :param solver: should be a string in 'auto' 'gradient_descent' 'coordinate_descent' 'lbfgs'
    :param tolerance: should be a float Object(python built-in)
    :param silent_mode: should be a boolean Object(python built-in), True to turn on the silent mode, and vice versa
    :return: model Object
    """
    start_timestamp = time.time()
    dataset = pd.DataFrame(dataset)

    feature_matrix_data = dataset.loc[:, features]
    weights = np.zeros(len(features)) if initial_weights is None else np.array(initial_weights)
    output = dataset[target]

    training_features = features
    costs = 0.

    if "(constant)" not in list(feature_matrix_data.columns):
        feature_matrix_data['(constant)'] = 1.
        training_features += ['(constant)']
        weights = np.append(weights, [0.])

    if solver == 'auto':
        raise NoImplementationError("No Implementation yet!")

    if l1_penalty != 0.:
        weights = regression.coordinate_descent(feature_matrix_data,
                                                weights,
                                                output,
                                                max_iteration,
                                                tolerance,
                                                l1_penalty,
                                                l2_penalty,
                                                silent_mode)
    elif solver == 'least_square':
        weights, costs = regression.lr_least_square(feature_matrix_data, output, l2_penalty)
    elif solver == "gradient_descent":
        weights, costs = regression.gradient_descent(feature_matrix_data,
                                                     output,
                                                     weights,
                                                     regression.lr_cost_function,
                                                     regression.lr_derivative,
                                                     step_size,
                                                     max_iteration, tolerance,
                                                     l2_penalty,
                                                     silent_mode)
    elif solver == "lbfgs":
        weights, costs, max_iteration = utilities.l_bfgs(training_features,
                                                         output,
                                                         weights,
                                                         regression.lr_cost_function,
                                                         lambda a, b, c, d: regression.lr_derivative(b, a, c, d),
                                                         iter_times=max_iteration,
                                                         l2_penalty=l2_penalty,
                                                         silent_mode=silent_mode
                                                         )
    else:
        raise InvalidParamError("Akane do not understand parameter solver=%s" % solver)

    weights = pd.Series(weights, index=training_features)

    costs = np.array(costs) if type(costs) == list else costs
    elapse = time.time() - start_timestamp

    profile = {
        dic['NUM_OF_EX']: feature_matrix_data.shape[0],
        dic['NUM_OF_FE']: len(features),
        dic['NUM_OF_CO']: len(weights),
        dic['SOLVER']: solver,
    }
    details = {
        dic['ITER']: 1 if solver == 'least_square' else max_iteration,
        dic['EPS_TIME']: elapse,
        dic['T_RSS']: costs,
    }
    if l1_penalty != 0.:
        details[dic['L1_P']] = l1_penalty

    if l2_penalty != 0:
        details[dic['L2_P']] = l2_penalty

    message.model_reporter(
        silent_mode, "Linear Regression", profile, details)

    model = models.LinearRegressionModel(profile, details, weights)

    return model


def logistic_classifier(dataset,
                        target,
                        feature,
                        initial_weights=None,
                        l1_penalty=0.,
                        l2_penalty=0.,
                        step_size=0.1,
                        max_iterations=500,
                        threshold=0.5,
                        validation_set=None,
                        solver='auto',
                        silent_mode=False,
                        data_weights=None, ):
    start_timestamp = time.time()
    dataset = pd.DataFrame(dataset)

    training_data = dataset.loc[:, feature]
    initial_weights = np.zeros(len(feature)) if initial_weights is None else np.array(initial_weights)
    output = dataset[target]
    if data_weights is None:
        data_weights = np.ones(len(dataset)) / len(dataset)

    training_feature = feature
    cost = 0.

    if "(constant)" not in list(training_data.columns):
        training_data['(constant)'] = 1.
        training_feature += ['(constant)']
        initial_weights = np.append(initial_weights, [0.])

    if solver == "gradient_descent":
        weights, cost = regression.gradient_descent(training_data,
                                                    output,
                                                    initial_weights,
                                                    lambda a, b, c, d: classification.lr_cost_function(a, b, c, d,
                                                                                                       data_weights=data_weights),
                                                    lambda a, b, c, d: classification.lr_derivatives(a, b, c, d,
                                                                                                     data_weights=data_weights),
                                                    step_size=step_size,
                                                    iter_times=max_iterations,
                                                    silent_mode=silent_mode,
                                                    l2_penalty=l2_penalty, )
    elif solver == 'lbfgs':
        weights, cost, max_iterations = utilities.l_bfgs(training_data,
                                                         output,
                                                         initial_weights,
                                                         lambda a, b, c, d: classification.lr_cost_function(a, b, c, d,
                                                                                                            data_weights=data_weights),
                                                         lambda a, b, c, d: classification.lr_derivatives(a, b, c, d,
                                                                                                          data_weights=data_weights),
                                                         iter_times=max_iterations,
                                                         l2_penalty=l2_penalty,
                                                         silent_mode=silent_mode,
                                                         )
    elif solver == 'auto':
        weights, cost = 0, 0
        pass
    else:
        weights, cost = 0, 0

    cost = np.array(cost) if type(cost) == list else cost
    elapse = time.time() - start_timestamp

    weights = pd.Series(weights, index=training_feature)

    profile = {
        dic['NUM_OF_EX']: training_data.shape[0],
        dic['NUM_OF_FE']: len(feature),
        dic['NUM_OF_CO']: len(initial_weights),
        dic['SOLVER']: solver,
    }
    detail = {
        dic['ITER']: max_iterations,
        dic['EPS_TIME']: elapse,
        dic['T_RSS']: cost,
        dic['THRESHOLD']: threshold,
    }
    if l2_penalty != 0:
        detail[dic['L2_P']] = l2_penalty

    model = models.LogisticRegressionModel(profile, detail, weights, threshold, training_feature)

    detail[dic['T_RSS']] = cost[-1] if len(cost) > 0 else "None"

    message.model_reporter(silent_mode, "Logistic Regression Model", profile, detail)
    return model


# Nearest Neighbour
def nearest_neighbor(dataset,
                     feature,
                     method,
                     distance=utilities.euclidean_distance,
                     silent_mode=False):
    start_timestamp = time.time()
    profile = {
        dic['NUM_OF_EX']: len(dataset),
        dic['NUM_OF_FE']: len(feature),
        dic['METHOD']: method,
        dic['DIS']: distance.__name__,
    }
    elapse = time.time() - start_timestamp
    details = {
        dic["EPS_TIME"]: elapse,
    }

    model = models.NearestNeighbourModel(profile, details, dataset, feature, distance, method)
    message.model_reporter(silent_mode, "Nearest Neighbour", profile, details)
    return model


def kernel_regression(dataset, features, target, kernel, silent_mode):
    start_timestamp = time.time()
    feature_matrix = dataset.loc[:, features]
    target_vector = dataset.loc[:, target]

    elapse = time.time() - start_timestamp
    profile = {
        dic['NUM_OF_EX']: len(dataset),
        dic['NUM_OF_FE']: len(features),
    }
    detail = {
        dic['EPS_TIME']: elapse,
    }

    model = models.KernelRegressionModel(profile, detail, feature_matrix, target_vector, kernel)
    message.model_reporter(silent_mode, "Kernel Regression", profile, detail)

    return model


def decision_tree(dataset,
                  features,
                  target,
                  weights=None,
                  method='id3',
                  silent_mode=False):
    start_timestamp = time.time()

    if weights is None:
        weights = np.ones(len(dataset))

    tree = classification.decision_tree_create(dataset, features, target, method, weights=weights)

    elapse = time.time() - start_timestamp
    profile = {
        dic['NUM_OF_EX']: len(dataset),
        dic['NUM_OF_FE']: len(features),
        dic['METHOD']: method,
    }
    detail = {
        dic['EPS_TIME']: elapse,
    }
    message.model_reporter(silent_mode, "Decision Tree", profile, detail)
    model = models.DecisionTreeModel(profile, detail, tree)

    return model


def adaboost(dataset,
             features,
             target,
             basic_classifier,
             num_of_classifier,
             initial_weights=None,
             silent_mode=False):
    start_timestamp = time.time()

    if initial_weights is None:
        initial_weights = np.ones(len(dataset)) / len(dataset)

    classifiers, weights = classification.adaboost_train(basic_classifier,
                                                         dataset,
                                                         features,
                                                         target,
                                                         initial_weights,
                                                         num_of_classifier,
                                                         silent_mode)

    elapse = time.time() - start_timestamp
    profile = {
        dic['NUM_OF_EX']: len(dataset),
        dic['NUM_OF_FE']: len(features),
        dic['NUM_OF_CF']: num_of_classifier,
    }
    details = {
        dic['EPS_TIME']: elapse,
        dic['CF']: basic_classifier.__name__,
    }
    message.model_reporter(silent_mode, "AdaBoost", profile, details)
    model = models.AdaboostModel(profile, details, weights, classifiers)

    return model


# data@param should be a matrix
# target@param should be a vector
# architecture@param imply the # of neural unit in each layer (from input layer to output layer)
def neural_networks(data,
                    target,
                    architecture,
                    init_weights=None,
                    activation=utilities.sigmoid,
                    activation_grad=utilities.sigmoid_gradient,
                    solver='lbfgs',
                    data_weights=None,
                    max_iteration=300,
                    l2_penalty=0.,
                    silent_mode=False):
    start_timestamp = time.time()
    if init_weights is None:
        weights = classification.nn_weight_init(architecture)
    else:
        weights = init_weights

    data = np.array(data)
    # add bias te
    data = np.c_[np.ones(len(data)), data]
    # encode target
    onehot_target = utilities.one_hot_encoder(target, architecture[-1])
    rolling_weights = classification.nn_rolling(weights)

    if solver == 'lbfgs':
        rolling_weights, costs, max_iteration = utilities.l_bfgs(data,
                                                                 onehot_target,
                                                                 rolling_weights,
                                                                 lambda a, b, c, d: classification.nn_cost_function(a,
                                                                                                                    b,
                                                                                                                    c,
                                                                                                                    activation,
                                                                                                                    architecture,
                                                                                                                    l2_penalty=d),
                                                                 lambda a, b, c, d: classification.nn_gradient(a,
                                                                                                               b,
                                                                                                               c,
                                                                                                               activation,
                                                                                                               activation_grad,
                                                                                                               architecture,
                                                                                                               l2_penalty=d),
                                                                 l2_penalty=l2_penalty,
                                                                 silent_mode=silent_mode
                                                                 )
    elif solver == 'gradient_descent':
        pass
    elif solver == "auto":
        pass
    else:
        raise Exception
    elapse = time.time() - start_timestamp

    profile = {
        dic['NUM_OF_EX']: len(data),
        dic['NUM_OF_LAYER']: len(architecture),
        dic['SOLVER']: solver,
    }
    detail = {
        dic['ITER']: max_iteration,
        dic['EPS_TIME']: elapse,
        dic['T_RSS']: costs,
    }
    model = models.NeuralNetworkModel(profile, detail, weights, architecture, activation)

    # detail[dic['T_RSS']] = costs[-1] if type(costs) == list else costs
    message.model_reporter(silent_mode, "Neural Networks", profile, detail)

    return model


def kmeans(data,
           features,
           k,
           distance_func=utilities.euclidean_distance,
           initial_centroids=None,
           initial_method='random',
           heterogeneity=None,
           max_iteration=500,
           silent_mode=False
           ):
    start_timestamp = time.time()

    data = pd.DataFrame(data)

    feature_matrix = data.loc[:, features]

    if initial_centroids is None:
        initial_centroids = clustering.initialize_centroid(feature_matrix,
                                                           k,
                                                           method=initial_method,
                                                           distance_func=distance_func)

    centroids, assignments, max_iteration = clustering.kmeans_train(feature_matrix,
                                                                    k,
                                                                    distance_func,
                                                                    initial_centroids=initial_centroids,
                                                                    heterogeneity_record=heterogeneity,
                                                                    max_iteration=max_iteration,
                                                                    silent_mode=silent_mode, )

    elapse = time.time() - start_timestamp
    profile = {
        dic['K']: k,
        dic['DIST']: distance_func.__name__,
    }
    detail = {
        dic['ITER']: max_iteration,
        dic['EPS_TIME']: elapse,
    }

    message.model_reporter(silent_mode, "K-Means", profile, detail)

    model = models.KMeansModel(profile, detail, k, centroids, assignments, heterogeneity, distance_func)
    return model


def gaussian_mixure_model(data,
                          features,
                          k,
                          initial_weights=None,
                          initial_mu=None,
                          initial_cov=None,
                          initial_method=None,
                          threshold=1e-4,
                          max_iteration=500,
                          silent_mode=False):
    start_timestamp = time.time()

    data = pd.DataFrame(data)
    feature_matrix = np.array(data.loc[:, features])

    if initial_weights is None:
        initial_weights = clustering.gmm_init_weight(feature_matrix, k, initial_method)

    if initial_mu is None:
        initial_mu = clustering.gmm_init_mu(feature_matrix, k, initial_method)

    if initial_cov is None:
        initial_cov = clustering.gmm_init_cov(feature_matrix, k, initial_method)

    result = clustering.em_for_gmm(feature_matrix,
                                   initial_mu,
                                   initial_cov,
                                   initial_weights,
                                   max_iteration,
                                   threshold,
                                   silent_mode)

    elapse = time.time() - start_timestamp

    profile = {
        dic['NUM_OF_EX']: len(data),
        dic['NUM_OF_FE']: len(features),
        dic['K']: k
    }
    detail = {
        dic['ITER']: max_iteration,
        dic['EPS_TIME']: elapse,
        dic['THRESHOLD']: threshold,
    }

    model = models.GaussianMixureModel(profile,
                                       detail,
                                       result['weights'],
                                       result['means'],
                                       result['covs'],
                                       result['loglik'],
                                       result['resp'])

    message.model_reporter(silent_mode, "Gaussian Mixure Model", profile, detail)

    return model
