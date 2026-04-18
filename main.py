import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, polygon, rectangle

from descriptors import extract_descriptor
from experiments import (
    test_rotation,
    test_scaling,
    test_translation,
    test_gaussian_noise,
    test_salt_pepper_noise,
    plot_curve
)
from matching import euclidean_distance


def create_blank_canvas(size=256):
    return np.zeros((size, size), dtype=np.uint8)


def make_circle(size=256, radius=50):
    img = create_blank_canvas(size)
    rr, cc = disk((size // 2, size // 2), radius)
    img[rr, cc] = 1
    return img


def make_rectangle(size=256, height=80, width=120):
    img = create_blank_canvas(size)
    start = ((size - height) // 2, (size - width) // 2)
    rr, cc = rectangle(start=start, extent=(height, width), shape=img.shape)
    img[rr, cc] = 1
    return img


def make_triangle(size=256):
    img = create_blank_canvas(size)
    r = np.array([60, 190, 190])
    c = np.array([128, 70, 186])
    rr, cc = polygon(r, c, img.shape)
    img[rr, cc] = 1
    return img


def show_shapes(shapes, titles):
    plt.figure(figsize=(12, 4))
    for i, (shape, title) in enumerate(zip(shapes, titles), start=1):
        plt.subplot(1, len(shapes), i)
        plt.imshow(shape, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()


def compare_base_shapes(method="similitude"):
    circle = make_circle()
    rectangle = make_rectangle()
    triangle = make_triangle()

    shapes = [circle, rectangle, triangle]
    names = ["Circle", "Rectangle", "Triangle"]

    print("\nBase shape descriptor distances:")
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            d1 = extract_descriptor(shapes[i], method=method)
            d2 = extract_descriptor(shapes[j], method=method)
            dist = euclidean_distance(d1, d2)
            print(f"{names[i]} vs {names[j]}: {dist:.4f}")

    show_shapes(shapes, names)
    return circle, rectangle, triangle


def run_experiments():
    circle, rectangle, triangle = compare_base_shapes(method="similitude")

    angles = list(range(0, 181, 15))
    rotation_distances = test_rotation(triangle, angles, method="similitude")
    plot_curve(angles, rotation_distances,
               xlabel="Rotation Angle (degrees)",
               ylabel="Euclidean Distance",
               title="Triangle: Distance vs Rotation")

    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    scale_distances = test_scaling(rectangle, scales, method="similitude")
    plot_curve(scales, scale_distances,
               xlabel="Scale Factor",
               ylabel="Euclidean Distance",
               title="Rectangle: Distance vs Scale")

    shifts = [(0, 0), (10, 10), (20, -15), (-25, 30)]
    translation_distances = test_translation(circle, shifts, method="similitude")
    plot_curve(range(len(shifts)), translation_distances,
               xlabel="Shift Case Index",
               ylabel="Euclidean Distance",
               title="Circle: Distance vs Translation")

    sigmas = [0.01, 0.03, 0.05, 0.08, 0.10]
    gaussian_distances = test_gaussian_noise(triangle, sigmas, method="similitude")
    plot_curve(sigmas, gaussian_distances,
               xlabel="Gaussian Noise Sigma",
               ylabel="Euclidean Distance",
               title="Triangle: Distance vs Gaussian Noise")

    amounts = [0.01, 0.03, 0.05, 0.08, 0.10]
    sp_distances = test_salt_pepper_noise(rectangle, amounts, method="similitude")
    plot_curve(amounts, sp_distances,
               xlabel="Salt & Pepper Amount",
               ylabel="Euclidean Distance",
               title="Rectangle: Distance vs Salt-and-Pepper Noise")


if __name__ == "__main__":
    run_experiments()