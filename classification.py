import numpy as np
import strings
import message
from math import sqrt
import pandas as pd
import utilities


def lr_cost_function(feature_matrix, weights, target, l2_penalty, data_weights=None):
    if data_weights is None:
        data_weights = np.ones(len(feature_matrix))

    feature_matrix = np.array(feature_matrix, dtype='float64')
    target = np.array(target, dtype='float64').ravel()
    weights = np.array(weights, dtype='float64').ravel()
    data_weights = np.array(data_weights, dtype='float64').ravel()

    m = feature_matrix.shape[0]

    scores = np.dot(feature_matrix, weights)
    predictions = utilities.sigmoid(scores)
    errors = target * np.log(predictions) + (1. - target) * np.log(1. - predictions)

    cost = -np.dot(data_weights, errors) + np.dot(weights, weights) * l2_penalty / (2. * m)

    return cost


def lr_derivatives(feature_matrix, weights, output, l2_penalty, data_weights=None):
    if data_weights is None:
        data_weights = np.ones(len(feature_matrix))

    feature_matrix = np.array(feature_matrix, dtype='float64')
    output = np.array(output, dtype='float64')
    weights = np.array(weights, dtype='float64')
    m = feature_matrix.shape[0]

    scores = np.dot(feature_matrix, weights)
    predictions = utilities.sigmoid(scores)
    errors = predictions - output
    errors = errors * data_weights

    derivatives = np.dot(errors, feature_matrix)
    derivatives = derivatives.ravel()
    derivatives = np.append(derivatives[:-1] + l2_penalty * weights[:-1] / m, derivatives[-1])

    return derivatives.ravel()


# data should be in type(DataFrame)
# Note that it return a list contains all classes,
# using [] to indexing items in the return set even it only contains 1 item.
def class_counter(data):
    data = pd.DataFrame(data)

    features = list(data.columns)
    sets = []
    for feature in features:
        col = data[feature]
        s = set()
        for item in col:
            s.add(item)
        sets.append(s)
    return sets


def intermediate_node_num_mistakes(labels, target_set):
    if len(labels) == 0:
        return 0

    labels = np.array(labels)

    cnt = []
    for target in target_set:
        cnt.append(
            len(labels[labels == target])
        )

    num_max_class = max(cnt)
    return sum(cnt) - num_max_class


def entropy(data, weights=None):
    if weights is None:
        weights = np.ones(len(data))

    classes = class_counter(data)[0]
    data = np.array(data)

    m = len(data)
    probs = []

    for c in classes:
        prob = len(data[data == c]) / m
        w = np.sum(weights[data == c]) / len(data[data == c])
        probs.append(prob * w)

    probs = np.array(probs)
    ent = - np.dot(probs, np.log(probs))
    return ent


def best_splitting_feature(data, features, target, method, weights=None):
    if weights is None:
        weights = np.ones(len(data))

    best_feature = None
    best_gain = float('-inf')

    num_data_points = len(data)

    if method == 'id3':
        empirical_entropy = entropy(data[target], weights=weights)

        for feature in features:
            d = data[[feature, target]]
            classes = class_counter(data[feature])[0]
            sum_conditional_entropy = 0.
            for c in classes:
                conditional_entropy = entropy(d[d[feature] == c][target], weights=weights[np.array(d[feature] == c)])
                sum_conditional_entropy += conditional_entropy

            gain = empirical_entropy - sum_conditional_entropy

            if gain > best_gain:
                best_gain = gain
                best_feature = feature

    return best_feature


def create_leaf(target_values, weights=None):
    leaf = {
        # splitting feature name
        "splitting_feature": None,
        # is leaf or not
        "is_leaf": True,
        # feature names
        "features": None,
        # child nodes
        "child": None
    }
    classes = list(class_counter(target_values)[0])

    target_values = np.array(target_values)
    probs = list()
    m = len(target_values)
    for c in classes:
        prob = len(target_values[target_values == c]) / m
        w = np.sum(weights[target_values == c]) / (len(target_values[target_values == c]))
        probs.append(prob * w)

    # class name in current leaf node
    leaf['classes'] = classes

    # class-wise probability in current leaf node
    leaf['prob'] = probs

    # final prediction
    leaf['prediction'] = classes[probs.index(max(probs))]
    return leaf


def decision_tree_create(data, features, target, method='id3', weights=None, current_depth=0., max_depth=10,
                         silent_mode=False):
    if weights is None:
        weights = np.ones(len(data))

    remaining_features = features[:]
    target_values = data[target]

    reporter = message.Reporter(silent_mode)

    reporter.report('--------------------------------------------------------------------')
    reporter.report("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

    if intermediate_node_num_mistakes(data[target], class_counter(data[target])[0]) == 0:
        reporter.report("Creating leaf node (same class)")
        return create_leaf(target_values, weights=weights)

    if not remaining_features:
        reporter.report("Creating leaf node (no remaining feature)")
        return create_leaf(target_values, weights=weights)

    if current_depth >= max_depth:
        reporter.report("Creating leaf node (max depth)")
        return create_leaf(target_values, weights=weights)

    splitting_feature = best_splitting_feature(data, remaining_features, target, method, weights=weights)
    classes = list(class_counter(data[splitting_feature])[0])
    splits = []
    split_weights = []
    for c in classes:
        split = data[data[splitting_feature] == c]
        split_weight = weights[np.array(data[splitting_feature]) == c]
        splits.append(split)
        split_weights.append(split_weight)

    remaining_features.remove(splitting_feature)
    reporter.report("split on feature %s. (%s)" % (splitting_feature, tuple(list(map(len, splits)))))

    for split in zip(splits, split_weights):
        if len(split[0]) == len(data):
            reporter.report("Creating leaf node")
            return create_leaf(split[0][target], weights=split[1])

    child_trees = []
    for split in zip(splits, split_weights):
        child_trees.append(
            decision_tree_create(split[0],
                                 remaining_features,
                                 target,
                                 method,
                                 current_depth=current_depth + 1,
                                 max_depth=max_depth,
                                 weights=split[1])
        )

    return {
        # splitting feature name
        "splitting_feature": splitting_feature,
        # is leaf or not
        "is_leaf": False,
        # child-wise class name
        "features": classes,
        # subtree in the order with feature (above)
        "child": child_trees,
        "prediction": None,
        "probs": None,
        "classes": None
    }


def weighted_mistakes(prediction, truth, weights=None):
    sum_of_weight = weights[np.array(prediction - truth) != 0].sum()
    return sum_of_weight


def classifier_weight(weighted_mistake):
    weight = np.log((1 - weighted_mistake) / weighted_mistake) / 2
    return weight


# basic classifier should be a lambda function where
# #1 parameter should be data
# #2 parameter should be features
# #3 parameter should be target
# #4 parameter should be data weight
def adaboost_train(basic_classifier, data, features, target, initial_weights, num_of_classifier, silent_mode=False):
    reporter = message.Reporter(silent_mode)

    # classifier data_weights
    alpha = []

    # data weights
    data_weights = initial_weights

    # list of classifiers
    classifiers = []

    target_values = np.array(data[target])

    for i in range(num_of_classifier):
        reporter.report('------------------------------------------------')
        reporter.report('Adaboost Iteration %d' % i)
        reporter.report('------------------------------------------------')

        classifier = basic_classifier(data, features, target, data_weights)
        predictions = classifier.classify(data, output_type='class')
        error = weighted_mistakes(predictions, target_values, data_weights)
        classifier_weights = classifier_weight(error)

        alpha.append(classifier_weights)
        classifiers.append(classifier)

        adjustment = np.array([
                                  np.exp(-classifier_weights) if item[0] == item[1]
                                  else np.exp(classifier_weights)
                                  for item in zip(predictions, target_values)
                                  ])

        data_weights = data_weights * adjustment
        normalizer = data_weights.sum()
        # print("predcition",predictions)
        print("error", error)
        # print("wei_cl", classifier_weights)
        print("data_weights:", data_weights.sum())
        # print("ad:", adjustment)
        data_weights = data_weights / normalizer

    return classifiers, np.array(alpha)


def nn_weight_init(architecture, seed=None):
    epsilon = np.sqrt(6) / np.sqrt(architecture[0] + architecture[-1])
    if seed is not None:
        np.random.seed(seed)

    weights = []
    for i, num in enumerate(architecture[:-1]):
        rows = architecture[i + 1] + 1 if i < len(architecture) - 2 else architecture[i + 1]
        cols = num + 1
        weight = np.random.rand(rows, cols) * 2 * epsilon - epsilon
        weights.append(weight)

    return weights


def nn_forward_propagation(data, weights, activation, architecture, output=None):
    weights = nn_unrolling(weights, architecture)
    a = data
    a_s = [data]
    z_s = [data]
    for weight in weights:
        z = np.dot(a, weight.T)
        a = activation(z)
        a_s.append(a)
        z_s.append(z)

    if output == 'full':
        return a_s, z_s
    return a_s


def nn_cost_function(data, weights, output, activation, architecture, l2_penalty=0., data_weights=None):
    if data_weights is None:
        data_weights = np.ones(len(data)) / len(data)

    prediction = nn_forward_propagation(data, weights, activation, architecture)[-1]

    cost = -np.dot(
        data_weights,
        (output * np.log(prediction[-1]) + (1 - output) * np.log(1 - prediction[-1])).sum(axis=1)
    )

    weights = nn_unrolling(weights, architecture)
    # add cost of regularization term
    for weight in weights:
        cost += (weight[:, 1:] * weight[:, 1:]).sum() * l2_penalty / 2.

    return cost


def nn_gradient(data, weights, output, activation, activation_grad, architecture, l2_penalty=0., data_weights=None):
    if data_weights is None:
        data_weights = np.ones(len(data)) / len(data)

    a_s, z_s = nn_forward_propagation(data, weights, activation, architecture, output='full')
    weights = nn_unrolling(weights, architecture)

    # initialize gradients of weights
    weights_grad = []
    for weight in weights:
        grad = np.zeros(weight.shape)
        weights_grad.append(grad)

    dets = [a_s[-1] - output]

    # compute delta for every layer
    for z, weight in zip(reversed(z_s[:-1]), reversed(weights)):
        det = np.dot(dets[-1], weight) * activation_grad(z)
        dets.append(det)

    dets = (dets[::-1])[1:]

    for i in range(len(weights_grad)):
        weights_grad[i] += np.dot(dets[i].T, a_s[i])
        penalty = l2_penalty * weights[i]

        # add regularization
        weights_grad[i][:, 1:] = (weights_grad[i][:, 1:] + penalty[:, 1:])
        weights_grad[i] /= len(data)

    return nn_rolling(weights_grad)


def nn_rolling(weights):
    rnt = np.array([])
    for weight in weights:
        rnt = np.append(rnt, weight.ravel())
    return rnt


def nn_unrolling(vector, architecture):
    rnt = []
    for i, num in enumerate(architecture[:-1]):
        rows = architecture[i + 1] + 1 if i < len(architecture) - 2 else architecture[i + 1]
        cols = num + 1
        weight = vector[:rows * cols].reshape((rows, cols))
        rnt.append(weight)
        vector = vector[rows * cols:]

    return rnt

