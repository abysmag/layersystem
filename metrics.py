# Metrics
# https://github.com/hpicgs/topic-models-and-dimensionality-reduction-sensitivity-study
import numpy as np
from scipy import spatial, stats
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score
def get_squared_distances_if_necessary(D_high_l, D_low_l):
    if isinstance(D_high_l, list) or len(D_high_l.shape) == 1:
        D_high = spatial.distance.squareform(D_high_l)
        D_low = spatial.distance.squareform(D_low_l)
    else:
        D_high = D_high_l
        D_low = D_low_l
    return D_high, D_low

def metric_trustworthiness(X_high, X_low, D_high_m, D_low_m, k=7):
    D_high, D_low = get_squared_distances_if_necessary(D_high_m, D_low_m)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    try:
        trustworthiness = float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())
    except AttributeError:  # Everything stayed constant
        trustworthiness = 1.0

    return trustworthiness


def metric_continuity(X_high, X_low, D_high_l, D_low_l, k=7):
    D_high, D_low = get_squared_distances_if_necessary(D_high_l, D_low_l)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    try:
        continuity = float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())
    except AttributeError:  # Everything stayed the same
        continuity = 1.0

    return continuity

def metric_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]

def metric_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low) ** 2) / np.sum(D_high ** 2)

def metric_pearson_correlation(D_scatter1, D_scatter2):
    if len(D_scatter1.shape) == 1:
        return stats.pearsonr(D_scatter1, D_scatter2)[0]
    else:
        corrs = []
        for i in range(D_scatter1.shape[1]):
            corrs.append(stats.pearsonr(D_scatter1[:, i], D_scatter2[:, i])[0])
        return np.array(corrs)

def metric_spearman_correlation(D_scatter1, D_scatter2):
    if len(D_scatter1.shape) == 1:
        return stats.spearmanr(D_scatter1, D_scatter2)[0]
    else:
        corrs = []
        for i in range(D_scatter1.shape[1]):
            corrs.append(stats.spearmanr(D_scatter1[:, i], D_scatter2[:, i])[0])
        return np.array(corrs)

def metric_silhouette(X, labels):
    return silhouette_score(X, labels)

def metric_mse(X, X_hat):
    return np.mean(np.square(X - X_hat))

def metric_cluster_ordering(x_low1, x_low2, y):
    clf_scatter1 = NearestCentroid().fit(X=x_low1, y=y)
    clf_scatter2 = NearestCentroid().fit(X=x_low2, y=y)

    distance_list1 = compute_distance_list(clf_scatter1.centroids_)
    distance_list2 = compute_distance_list(clf_scatter2.centroids_)

    return metric_spearman_correlation(distance_list1, distance_list2)

def compute_distance_list(X, eval_distance_metric='euclidean'):
    return spatial.distance.pdist(X, eval_distance_metric)

