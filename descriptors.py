import numpy as np
from skimage.measure import moments, moments_central, moments_normalized, moments_hu

# Compute Hu moments (7-dimensional, invariant to rotation/scale/translation).
def hu_moment_descriptor(binary):
    image = binary.astype(float)
    mu = moments_central(image)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    # log transform for numerical stability
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    return hu

# Similitude moment descriptor (invariant to translation, rotation, and scale); computes normalized central moments up to order 3, results in 7-element vector; equivalent to Hu moments but derived explicitly from normalized central moments.
def similitude_moment_descriptor(binary):
    image = binary.astype(float)
    if image.sum() == 0:
        return np.zeros(7)

    # Raw moments
    m = moments(image, order=3)
    m00 = m[0, 0]
    if m00 == 0:
        return np.zeros(7)

    # Centroid
    cx = m[1, 0] / m00
    cy = m[0, 1] / m00

    # Central moments up to order 3
    mu = moments_central(image, center=(cx, cy), order=3)

    # Normalised central moments
    def nu(p, q):
        denom = m00 ** (1 + (p + q) / 2.0)
        return mu[p, q] / denom if denom != 0 else 0.0

    n20 = nu(2, 0)
    n02 = nu(0, 2)
    n11 = nu(1, 1)
    n30 = nu(3, 0)
    n12 = nu(1, 2)
    n21 = nu(2, 1)
    n03 = nu(0, 3)

    # Hu's 7 rotation-invariant moments
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = ((n30 - 3 * n12) * (n30 + n12) *
          ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
          (3 * n21 - n03) * (n21 + n03) *
          (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h6 = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) +
          4 * n11 * (n30 + n12) * (n21 + n03))
    h7 = ((3 * n21 - n03) * (n30 + n12) *
          ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
          (n30 - 3 * n12) * (n21 + n03) *
          (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))

    hu = np.array([h1, h2, h3, h4, h5, h6, h7])

    # Log transform for numerical stability
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    return hu

def extract_descriptor(binary, method="similitude"):
    if method == "similitude":
        return similitude_moment_descriptor(binary)
    elif method == "hu":
        return hu_moment_descriptor(binary)
    else:
        raise ValueError(f"Unknown descriptor method: {method}")