import numpy as np
from skimage.measure import moments, moments_central, moments_normalized, moments_hu


def hu_moment_descriptor(binary):
    image = binary.astype(float)

    mu = moments_central(image)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)

    # log transform (important for stability)
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    return hu


def similitude_moment_descriptor(binary):
    """
    Replace this later with your HW4 code.
    For now, we use Hu moments so everything runs.
    """
    return hu_moment_descriptor(binary)


def extract_descriptor(binary, method="similitude"):
    if method == "similitude":
        return similitude_moment_descriptor(binary)
    elif method == "hu":
        return hu_moment_descriptor(binary)
    else:
        raise ValueError("Unknown descriptor method")