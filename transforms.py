import numpy as np
from skimage import transform, util
from scipy.ndimage import shift as ndi_shift
from skimage import morphology
 
 
def rotate_image(binary, angle):
    """Rotate binary image by given angle (degrees)."""
    rotated = transform.rotate(
        binary.astype(float),
        angle=angle,
        resize=False,
        order=0,
        preserve_range=True
    )
    return (rotated > 0.5).astype(np.uint8)
 
 
def scale_image(binary, scale_factor):
    """Scale binary image by given factor, keeping the same canvas size."""
    h, w = binary.shape
    scaled = transform.rescale(
        binary.astype(float),
        scale=scale_factor,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )
    scaled = (scaled > 0.5).astype(np.uint8)
    new_h, new_w = scaled.shape
    output = np.zeros((h, w), dtype=np.uint8)
    if new_h <= h and new_w <= w:
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        output[start_y:start_y + new_h, start_x:start_x + new_w] = scaled
    else:
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        output = scaled[start_y:start_y + h, start_x:start_x + w]
    return output
 
 
def translate_image(binary, shift_y, shift_x):
    """Translate binary image by (shift_y, shift_x) pixels."""
    shifted = ndi_shift(binary.astype(float), shift=(shift_y, shift_x), order=0, cval=0.0)
    return (shifted > 0.5).astype(np.uint8)
 
 
def add_gaussian_noise(binary, sigma=0.05):
    """Add Gaussian noise and re-threshold."""
    noisy = util.random_noise(binary.astype(float), mode='gaussian', var=sigma ** 2)
    return (noisy > 0.5).astype(np.uint8)
 
 
def add_salt_pepper_noise(binary, amount=0.05):
    """Add salt-and-pepper noise and re-threshold."""
    noisy = util.random_noise(binary.astype(float), mode='s&p', amount=amount)
    return (noisy > 0.5).astype(np.uint8)
 
 
def erode_boundary(binary, radius=2):
    """Simulate boundary damage by erosion."""
    selem = morphology.disk(radius)
    return morphology.erosion(binary.astype(bool), selem).astype(np.uint8)


def dilate_boundary(binary, radius=2):
    """Simulate segmentation error by dilation."""
    selem = morphology.disk(radius)
    return morphology.dilation(binary.astype(bool), selem).astype(np.uint8)
 
 
def blur_and_threshold(binary, sigma=2.0):
    """
    Blur the binary image with a Gaussian filter then re-threshold.
    Simulates the effect of a slightly out-of-focus or low-resolution image
    before segmentation, as described in the project spec.
    """
    from skimage.filters import gaussian, threshold_otsu
    blurred = gaussian(binary.astype(float), sigma=sigma)
    thresh = threshold_otsu(blurred)
    return (blurred > thresh).astype(np.uint8)