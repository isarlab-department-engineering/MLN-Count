import yaml
import numpy as np
import math
import os.path
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture
# from tensorflow.core.config.flags import config


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def find_local_maxima(heatmap, neighborhood_size, threshold, threshold_rel=None):

    coordinates = peak_local_max(heatmap, min_distance=neighborhood_size, threshold_abs=threshold,
                                 threshold_rel=threshold_rel, exclude_border=True)
    return coordinates


def clusterize_gmm(maps, n_clusters):

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    height, width = maps.shape
    X = maps.reshape((height * width, 1))
    gmm.fit(X)
    labels = gmm.predict(X)
    clustered_heatmap = labels.reshape((height, width))

    return clustered_heatmap


def TC_LMD(strategy, maps, n_clusters=5, min_peak_distance=2, content_score_th=0.2):

    # Clusterize with gaussian mixture model #n cluster to identify n regions
    clustered_heatmap = clusterize_gmm(maps, n_clusters)

    local_means = np.zeros(n_clusters, float)
    local_dev_stds = np.zeros(n_clusters, float)
    for j in range(n_clusters):
        # max = np.max(map_gauss[clustered_heatmap == j])
        mean = np.mean(maps[clustered_heatmap == j])
        dev_std = np.std(maps[clustered_heatmap == j])
        local_means[j] = mean
        local_dev_stds[j] = dev_std

    # Evaluate the global mean and standard deviation of the heatmap
    global_mean = np.mean(maps)
    global_std = np.std(maps)

    # Evaluate the standard deviation of the local means (of the clusters)
    local_dev_std = np.std(local_means)
    local_mean = np.mean(local_means)

    # We sum mean+std and compare it with the global mean: if they are comparable there is no information
    content_score = np.abs(global_mean - local_mean)

    threshold_peak = np.inf
    # If the content score > 1 it should be a fruit (activation map with information)
    if content_score > content_score_th:
        # Pick the cluster with more information with respect to each others
        cluster_id_max = np.argmax(abs(local_means + local_dev_std - global_mean))

        cluster_values = maps[clustered_heatmap == cluster_id_max]

        # Set the threshold
        if strategy == 'mean':
            threshold_peak = np.mean(cluster_values)
        else:
            threshold_peak = np.min(cluster_values)

    detec_coordinates = find_local_maxima(maps, threshold=threshold_peak,
                                          neighborhood_size=min_peak_distance)

    return detec_coordinates
