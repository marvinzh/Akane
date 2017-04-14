import numpy as np
import pandas as pd
import strings
import message
import utilities
import classification

dic = strings.MODELS_EN


class Model(object):
    """
    Super class for all Akane Model

    """

    def __init__(self, profile, detail):
        super(Model, self).__init__()
        self.type = dic["UM"]
        self.profile = profile
        self.details = detail

    def model_profile(self):
        message.model_reporter(False, self.type, self.profile, self.details)


class LinearRegressionModel(Model):
    """
    LinearRegressionModel
    """

    def __init__(self, profile, detail, weights):
        super(LinearRegressionModel, self).__init__(profile, detail)
        self.type = dic["LINEAR_M"]

        self.weights = weights
        self.features = list(self.weights.index)

    def predict(self, data):
        d = data.copy()
        d['(constant)'] = 1.

        features = list(self.weights.index)
        feature_matrix = np.array(d.loc[:, features])
        weights = np.array(self.weights)
        prediction = np.dot(feature_matrix, weights)

        return prediction

    def score(self, data, y):
    	prediction = self.predict(data)
    	errors = prediction - y
    	RSS = np.dot(errors,errors) / len(data)
    	return RSS


class NearestNeighbourModel(Model):
    """docstring for NearestNeighborModel"""

    def __init__(self, profile, detail, dataset, feature, distance_func, method):
        super(NearestNeighbourModel, self).__init__(profile, detail)
        self.type = dic["NN_M"]

        self.dataset = dataset
        self.feature = feature
        self.distance_func = distance_func
        self.method = method

    def query(self, data, label=None, k=5, radius=float('inf')):
        neighbours = pd.DataFrame(columns=[dic['QUERY_L'], dic['REF_L']], dtype='float64')

        if label is None:
            label = self.feature[0]

        neighbours[dic['REF_L']] = self.dataset[label]
        neighbours[dic['QUERY_L']] = data[label]
        neighbours.astype('float64')

        target = data[self.feature]
        dataset = self.dataset[self.feature]

        # distances = np.ones(len(dataset))
        if self.method == dic['NN_BF_METHOD']:
            distances = self.distance_func(target, dataset)
        elif self.method == dic['NN_LSH_METHOD']:
            distances = []
        elif self.method == dic['NN_AUTO_METHOD']:
            distances = []
        else:
            distances = []

        neighbours[dic['DIS_L']] = distances
        neighbours = neighbours.sort_values(by=dic['DIS_L'])
        neighbours[dic['RANK_L']] = range(1, len(neighbours) + 1)
        neighbours = neighbours[neighbours[dic['DIS_L']] <= radius].iloc[:k]
        # neighbours.index = range(len(neighbours))
        indexes = list(neighbours.index)

        return neighbours, self.dataset.iloc[indexes]

    def predict(self, data, target, k=5, radius=float('inf'), weight_func=None):

        neighbours = self.query(data, k=5, label='price', radius=radius)[1]

        neighbours_values = np.array(neighbours[target])
        weights = np.ones(k)

        if weight_func is not None:
            weights = map(weight_func, neighbours_values)

        prediction = np.dot(weights, neighbours_values) / weights.sum()

        return prediction


class KernelRegressionModel(Model):
    def __init__(self, profile, detail, feature_matrix, target, features, kernel):
        super(KernelRegressionModel, self).__init__(profile, detail)
        self.type = dic["KERNEL_M"]

        self.data_matrix = np.array(feature_matrix)
        self.target = np.array(target)
        self.kernel = kernel
        self.features = features

    def predict(self, dataset, kernel=None):
        data = dataset.loc[:,self.features]
        data = np.array(data, dtype='float64')
        if kernel is None:
            kernel = self.kernel

        target = self.target

        error = np.array(list(
            map(lambda x: x - self.data_matrix, data)
        ))

        weights = np.array(list(
            map(lambda x: kernel(x), error)
        ))

        # print(len(weights))
        denominator = weights.sum(axis=1)

        prediction = np.dot(weights, target) / denominator
        return prediction


class LogisticRegressionModel(Model):
    """
        Logistic Regression Model

    """

    def __init__(self, profile, detail, weights, threshold, feature):
        super(LogisticRegressionModel, self).__init__(profile, detail)
        self.type = dic["LOG_M"]

        self.weights = weights
        self.threshold = threshold
        self.feature = feature

    def classify(self, df, output_type=dic['TYPE_CLASS']):
        if "(constant)" not in list(df.columns):
            df['(constant)'] = 1.
        data = df.loc[:, self.feature]
        data = np.array(data, dtype='float64')

        scores = np.dot(data, self.weights)
        predictions = utilities.sigmoid(scores)
        classes = np.array(list(
            map(lambda x: 1 if x >= self.threshold else 0, predictions)
        ))
        if output_type == dic['TYPE_PROB']:
            return predictions
        elif output_type == dic['TYPE_CLASS']:
            return classes
        else:
            print('Invalid Output Type!')
            return
            # return predictions if output_type == dic['TYPE_PROB'] else classes


class DecisionTreeModel(Model):
    def __init__(self, profile, detail, tree):
        super(DecisionTreeModel, self).__init__(profile, detail)
        self.type = dic["DEC_M"]

        self.tree = tree

    def classify(self, data, output_type=dic['TYPE_CLASS']):
        data = pd.DataFrame(data)
        tree = self.tree
        # predictions = (list(
        #     map(lambda x: self.search(tree, pd.DataFrame(x).loc, output_type), data)
        # ))
        predictions = []
        for i in range(len(data)):
            prediction = self.search(tree, data.loc[i], output_type)
            predictions.append(prediction)

        # Note that the return type is list object if output_type=probability
        return np.array(predictions) if output_type == dic['TYPE_CLASS'] else predictions

    def search(self, tree, data, output_type=dic['TYPE_CLASS']):
        if tree['is_leaf']:
            if output_type == dic['TYPE_CLASS']:
                return tree['prediction']
            elif output_type == dic['TYPE_PROB']:
                return pd.Series(tree['prob'], index=tree['classes'])
            else:
                print('Invalid Output Type!')
                return
        else:
            # adding [0] to get intact string
            splitting_feature_value = str.strip(data[tree['splitting_feature']])
            features = tree['features']
            features = list(map(str.strip, features))
            child_index = features.index(splitting_feature_value)
            return self.search(tree['child'][child_index], data, output_type)


class AdaboostModel(Model):
    def __init__(self, profile, detail, weights, classifiers):
        super(AdaboostModel, self).__init__(profile, detail)
        self.type = dic['ADA_M']

        self.weights = weights
        self.classifiers = classifiers

    def classify(self, data, start=0, end=float('inf')):

        if end == float('inf') or end > len(self.classifiers):
            end = len(self.classifiers)

        start = 0 if start < 0 else start

        # threshold = self.weights[start:end].sum() / 2
        scores = np.zeros(len(data))

        for i in range(start, end):
            predictions = self.classifiers[i].classify(data, output_type='class')
            predictions = np.array(list(map(lambda x: 1 if x > 0 else -1, predictions)))
            scores = scores + self.weights[i] * predictions

        predictions = list(map(lambda x: 1 if x >= 0 else 0, scores))
        return np.array(predictions)


class NeuralNetworkModel(Model):
    def __init__(self, profile, detail, weights, architecture, activation):
        super(NeuralNetworkModel, self).__init__(profile, detail)
        self.type = dic["NEN_M"]

        self.weights = weights
        self.architecture = architecture
        self.activation = activation

    def classify(self, data, output_type=dic['TYPE_CLASS']):
        a_raw = classification.nn_forward_propagation(data, self.weights, self.activation, self.architecture)[-1]

        row_wise_max = a_raw.max(axis=1)

        predictions = np.array(list(
            map(lambda x: np.where(a_raw == x)[1][0], row_wise_max)
        ))
        if output_type == dic['TYPE_PROB']:
            df = pd.DataFrame([predictions, row_wise_max]).transpose()
            df.columns = ['prediction', 'probability']
            return df

        return predictions


class KMeansModel(Model):
    def __init__(self, profile, detail, k, centroids, assignments, heterogeneity, distance_func):
        super(KMeansModel, self).__init__(profile, detail)
        self.type = dic["KM_M"]

        self.k = k
        self.centroids = centroids
        self.assignments = assignments
        self.heterogeneity = heterogeneity
        self.distance_func = distance_func

    def assign(self, data):
        data = np.array(data)
        distance = self.distance_func(self.centroids, data)
        assignments = np.argmin(distance, axis=1)
        return assignments


class GaussianMixureModel(Model):
    def __init__(self, profile, detail, weights, means, covs, log_likelihood, responsibility):
        super(GaussianMixureModel, self).__init__(profile, detail)
        self.type = dic['GMM']

        self.weights = weights
        self.means = means
        self.covariances = covs
        self.log_likelihood = log_likelihood
        self.responsibility = responsibility
        self.k = len(means)

    def assign(self, data):
        assignments = []
        for i in range(self.k):
            assignment = multivariate_normal(data, self.means[i], self.covariances[i])
            assignments.append(assignment)

        return np.array(assignments)

    # if __name__ == '__main__':
    #     profile = [
    #         #  Num of data points
    #         1000,
    #         #  num of feature
    #         123,
    #         # num of input
    #         124,
    #         # solver
    #         "test solver",
    #     ]
    #     details = [
    #         # num of iteration
    #         123,
    #         #  elapse time
    #         123.456,
    #         # cost
    #         12345678,
    #     ]
    #     model = LinearRegressionModel([1, 2, 3], profile, details)
    #     # print(model.weights)
    # model.model_profile()
