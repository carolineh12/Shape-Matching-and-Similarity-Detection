import numpy as np

# compute Euclidean distance between two descriptor vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# return is a match or not and the distance
def is_match(vec1, vec2, threshold):
    distance = euclidean_distance(vec1, vec2)
    return distance <= threshold, distance

# compute classification threshold as midpoint between class means, returns threshold
def compute_threshold(same_distances, diff_distances):
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(diff_distances)
    threshold = 0.5 * (same_mean + diff_mean)
    return threshold

# compute overall binary classification accuracy
def compute_accuracy(same_distances, diff_distances, threshold):
    true_positives = np.sum(np.array(same_distances) <= threshold)
    true_negatives = np.sum(np.array(diff_distances) > threshold)
    total = len(same_distances) + len(diff_distances)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    return accuracy

# return a dict of statistics about matching performance
def compute_matching_stats(same_distances, diff_distances):
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