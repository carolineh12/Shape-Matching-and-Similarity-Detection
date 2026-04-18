import numpy as np
from skimage import color, filters, morphology, measure, util


def to_grayscale(image):
    """Convert RGB or grayscale image to float grayscale in [0, 1]."""
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float64)

    if gray.max() > 1.0:
        gray = gray / 255.0

    return gray


def binarize_image(gray, threshold=None):
    """Convert grayscale image to binary using Otsu or a provided threshold."""
    if threshold is None:
        threshold = filters.threshold_otsu(gray)
    binary = gray < threshold if np.mean(gray) > 0.5 else gray > threshold
    return binary.astype(np.uint8)


def clean_binary(binary, min_size=100):
    """Remove small objects/holes and apply light morphology."""
    mask = binary.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask.astype(np.uint8)


def largest_component(binary):
    """Keep only the largest connected component."""
    labels = measure.label(binary)
    props = measure.regionprops(labels)

    if not props:
        return binary.astype(np.uint8)

    largest_region = max(props, key=lambda region: region.area)
    mask = labels == largest_region.label
    return mask.astype(np.uint8)


def preprocess_image(image, threshold=None, min_size=100):
    """Full preprocessing pipeline."""
    gray = to_grayscale(image)
    binary = binarize_image(gray, threshold=threshold)
    cleaned = clean_binary(binary, min_size=min_size)
    component = largest_component(cleaned)
    return gray, binary, cleaned, component