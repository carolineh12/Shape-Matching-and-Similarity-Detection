import numpy as np


def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two descriptor vectors."""
    return np.linalg.norm(vec1 - vec2)


def is_match(vec1, vec2, threshold):
    """Return (match: bool, distance: float) given a threshold."""
    distance = euclidean_distance(vec1, vec2)
    return distance <= threshold, distance


def compute_threshold(same_distances, diff_distances):
    """
    Compute classification threshold as midpoint between class means.
    Returns threshold and simple stats.
    """
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(diff_distances)
    threshold = 0.5 * (same_mean + diff_mean)
    return threshold


def compute_accuracy(same_distances, diff_distances, threshold):
    """
    Given same-class and different-class distances and a threshold,
    compute overall binary classification accuracy.
    """
    true_positives = np.sum(np.array(same_distances) <= threshold)
    true_negatives = np.sum(np.array(diff_distances) > threshold)
    total = len(same_distances) + len(diff_distances)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    return accuracy


def compute_matching_stats(same_distances, diff_distances):
    """
    Return a dict of statistics about matching performance.
    """
    threshold = compute_threshold(same_distances, diff_distances)
    accuracy = compute_accuracy(same_distances, diff_distances, threshold)
    return {
        "same_mean": np.mean(same_distances),
        "same_std": np.std(same_distances),
        "diff_mean": np.mean(diff_distances),
        "diff_std": np.std(diff_distances),
        "threshold": threshold,
        "accuracy": accuracy,
    }