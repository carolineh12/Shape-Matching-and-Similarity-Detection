import numpy as np


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def is_match(vec1, vec2, threshold):
    distance = euclidean_distance(vec1, vec2)
    return distance <= threshold, distance


def compute_threshold(same_distances, diff_distances):
    """
    Simple threshold: midpoint between means.
    You can improve this later.
    """
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(diff_distances)
    return 0.5 * (same_mean + diff_mean)