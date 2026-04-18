import numpy as np
from skimage.measure import moments, moments_central, moments_normalized, moments_hu

def normalized_central_moment(mu_pq, mu00, p, q):
    gamma = 1.0 + (p + q) / 2.0
    return mu_pq / (mu00 ** gamma)


def central_moment(x_shifted, y_shifted, img, p, q):
    return np.sum((x_shifted ** p) * (y_shifted ** q) * img)


def similitude_moment_descriptor(binary):
    """
    Uses your HW4 similitude/Hu-style moment code on a binary image.
    Returns a 7D feature vector.
    """
    img = np.asarray(binary, dtype=np.float64)

    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    yy, xx = np.indices(img.shape)

    mu00 = np.sum(img)
    if mu00 == 0:
        return np.zeros(7)

    mu10 = np.sum(xx * img)
    mu01 = np.sum(yy * img)

    xbar = mu10 / mu00
    ybar = mu01 / mu00

    x = xx - xbar
    y = yy - ybar

    mu20 = central_moment(x, y, img, 2, 0)
    mu02 = central_moment(x, y, img, 0, 2)
    mu11 = central_moment(x, y, img, 1, 1)
    mu30 = central_moment(x, y, img, 3, 0)
    mu03 = central_moment(x, y, img, 0, 3)
    mu21 = central_moment(x, y, img, 2, 1)
    mu12 = central_moment(x, y, img, 1, 2)

    n20 = normalized_central_moment(mu20, mu00, 2, 0)
    n02 = normalized_central_moment(mu02, mu00, 0, 2)
    n11 = normalized_central_moment(mu11, mu00, 1, 1)
    n30 = normalized_central_moment(mu30, mu00, 3, 0)
    n03 = normalized_central_moment(mu03, mu00, 0, 3)
    n21 = normalized_central_moment(mu21, mu00, 2, 1)
    n12 = normalized_central_moment(mu12, mu00, 1, 2)

    phi1 = n20 + n02
    phi2 = (n20 - n02) ** 2 + 4 * (n11 ** 2)
    phi3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    phi4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    phi5 = (
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    )
    phi6 = (
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
        + 4 * n11 * (n30 + n12) * (n21 + n03)
    )
    phi7 = (
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    )

    features = np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7], dtype=np.float64)

    # log scaling helps comparisons stay numerically stable
    features = np.sign(features) * np.log1p(np.abs(features))

    return features


def extract_descriptor(binary, method="similitude"):
    if method == "similitude":
        return similitude_moment_descriptor(binary)
    else:
        raise ValueError("Unknown descriptor method")