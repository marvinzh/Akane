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
