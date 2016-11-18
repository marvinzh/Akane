import numpy as np
import utilities
import message


# data@param should be a feature matrix
def initialize_centroid(data, k, method='random', distance_func=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = np.array(data)
    n = data.shape[0]
    if method == "random":
        index = np.random.randint(0, n, k)
        centroids = np.array(data[index])
    elif method == 'kmeans++':
        if seed is not None:
            np.random.seed(seed)
        if distance_func is None:
            distance_func = utilities.euclidean_distance

        centroids = np.zeros((k, data.shape[1]))
        idx = np.random.randint(data.shape[0])
        centroids[0] = np.array(data[idx])

        distance = distance_func(centroids[0], data).ravel()

        for i in range(1, k):
            idx = np.random.choice(len(data), 1, p=distance / distance.sum())
            centroids[i] = data[idx]
            distance = np.min(np.array(list(
                map(lambda x: distance_func(x, data), centroids[:i + 1])
            )).T, axis=1)
    else:
        raise Exception

    return centroids


# centroids@param should be a 2-dim matrix
# distance_func@param should receive at least 2 parameter, target@param: a vector, source@param: a 2-dim matrix
def assign_cluster(data, centroids, distance_func):
    distances = np.array(list(
        map(lambda x: distance_func(x, data), centroids)
    )).T
    return np.argmin(distances, axis=1)


def revise_centroids(data, labels, k):
    centroids = []
    for i in range(k):
        cluster = data[labels == i]
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    return centroids


def compute_heterogeneity(data, centroids, labels, distance_func):
    k = len(centroids)
    heterogeneity = 0.
    for i in range(k):
        members = data[labels == i]
        distances = distance_func(centroids[i], members)
        squared_distance = np.dot(distances, distances)
        heterogeneity += np.sum(squared_distance)

    return heterogeneity


def kmeans_train(data,
                 k,
                 distance_func,
                 initial_centroids=None,
                 heterogeneity_record=None,
                 max_iteration=500,
                 silent_mode=False):
    reporter = message.Reporter(silent_mode)

    assignments = None
    if initial_centroids is None:
        initial_centroids = initialize_centroid(data, k)

    centroids = initial_centroids.copy()

    i = 0
    while i < max_iteration:
        reporter.report("Iteration %d" % i)
        cluster_assignments = assign_cluster(data, centroids, distance_func)

        if assignments is not None:
            change = np.sum(assignments != cluster_assignments)
            reporter.report("%d elements changed their cluster assignment." % change)
            if change == 0:
                reporter.report("No elements changed.")
                break

        assignments = cluster_assignments.copy()
        centroids = revise_centroids(data, cluster_assignments, k)

        if heterogeneity_record is not None:
            score = compute_heterogeneity(data, centroids, assignments, distance_func)
            heterogeneity_record.append(score)
        i += 1
        reporter.report("")
    return centroids, assignments, i

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])

    ll = 0
    for d in data:

        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)

    return ll


def gmm_init_weight(fature_matrix, k, init_method):
    if init_method is None:
        return np.ones(k) / k


def gmm_init_mu(feature_matrix, k, init_method):
    if init_method is None:
        # chosen = np.random.choice(feature_matrix.shape[0], k, replace=False)
        chosen = [0, 1, 2]
        mu = [feature_matrix[idx] for idx in chosen]
        return mu


def gmm_init_cov(feature_matrix, k, init_method):
    num_of_feature = feature_matrix.shape[1]
    if init_method is None:
        covs = [np.diag(np.ones(num_of_feature))] * k

        return np.array(covs)


def em_for_gmm(data, init_means, init_covariances, init_weights, max_iter=1000, threshold=1e-4, silent_mode=False):
    reporter = message.Reporter(silent_mode)

    # Make copies of initial parameters, which we will update during each iteration
    means = init_means.copy()
    covariances = init_covariances.copy()
    weights = init_weights.copy()

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)

    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]

    for i in range(max_iter):

        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        for j in range(num_data):
            for k in range(num_clusters):
                resp[j, k] = weights[k] * scipy.stats.multivariate_normal.pdf(data[j],
                                                                              mean=means[k],
                                                                              cov=covariances[k])

        row_sums = resp.sum(axis=1)[:, np.newaxis]
        # normalize over all possible cluster assignments
        resp = resp / row_sums

        # M-step
        for k in range(num_clusters):

            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights[k] = resp[:, k].sum() / float(num_data)

            # Update means for cluster k using the M-step update rule for the mean variables.
            weighted_sum = 0
            for j in range(num_data):
                weighted_sum += resp[j, k] * data[j]

            means[k] = weighted_sum / float(resp[:, k].sum())

            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                weighted_sum += resp[j, k] * np.outer(data[j] - means[k], data[j] - means[k])

            covariances[k] = weighted_sum / float(resp[:, k].sum())

        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        reporter.report("Iteration: %s  Log Likelihood: %s" % (i, ll_latest))

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < threshold and ll_latest > -np.inf:
            break
        ll = ll_latest

    out = {
        'weights': weights,
        'means': means,
        'covs': covariances,
        'loglik': ll_trace,
        'resp': resp
    }

    return out

# if __name__ == '__main__':
#     centroid = [
#         [1, 2],
#         [0, 0],
#         [2, 3],
#     ]
#     data = np.array([
#         [1, 2],
#         [2, 3],
#         [0, 0],
#         [19, 5]
#     ])
#     label = np.array([0, 1, 0])
#     # print(assign_cluster(data, centroid, utilities.euclidean_distance))
#     # print(revise_centroids(data, label, 2))
#     print(initialize_centroid(data, 2, method='kmeans++', seed=1))
