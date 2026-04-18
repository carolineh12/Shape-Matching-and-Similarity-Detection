import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocess_image, largest_component
from transforms import (
    rotate_image,
    scale_image,
    translate_image,
    add_gaussian_noise,
    add_salt_pepper_noise
)
from descriptors import extract_descriptor
from matching import euclidean_distance


def descriptor_distance(binary1, binary2, method="similitude"):
    d1 = extract_descriptor(binary1, method=method)
    d2 = extract_descriptor(binary2, method=method)
    return euclidean_distance(d1, d2)


def test_rotation(binary, angles, method="similitude"):
    distances = []
    for angle in angles:
        transformed = rotate_image(binary, angle)
        transformed = largest_component(transformed)
        dist = descriptor_distance(binary, transformed, method=method)
        distances.append(dist)
    return distances


def test_scaling(binary, scales, method="similitude"):
    distances = []
    for scale in scales:
        transformed = scale_image(binary, scale)
        transformed = largest_component(transformed)
        dist = descriptor_distance(binary, transformed, method=method)
        distances.append(dist)
    return distances


def test_translation(binary, shifts, method="similitude"):
    distances = []
    for shift_y, shift_x in shifts:
        transformed = translate_image(binary, shift_y, shift_x)
        transformed = largest_component(transformed)
        dist = descriptor_distance(binary, transformed, method=method)
        distances.append(dist)
    return distances


def test_gaussian_noise(binary, sigmas, method="similitude"):
    distances = []
    for sigma in sigmas:
        transformed = add_gaussian_noise(binary, sigma=sigma)
        transformed = largest_component(transformed)
        dist = descriptor_distance(binary, transformed, method=method)
        distances.append(dist)
    return distances


def test_salt_pepper_noise(binary, amounts, method="similitude"):
    distances = []
    for amount in amounts:
        transformed = add_salt_pepper_noise(binary, amount=amount)
        transformed = largest_component(transformed)
        dist = descriptor_distance(binary, transformed, method=method)
        distances.append(dist)
    return distances


def plot_curve(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()