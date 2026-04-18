import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
from preprocessing import largest_component
from transforms import (
    rotate_image, scale_image, translate_image,
    add_gaussian_noise, add_salt_pepper_noise,
    erode_boundary, dilate_boundary, blur_and_threshold,
)
from descriptors import extract_descriptor
from matching import euclidean_distance, compute_threshold, compute_matching_stats
 
 
# ─────────────────────────────────────────────
# Similarity
# ─────────────────────────────────────────────
 
def descriptor_distance(binary1, binary2, method="similitude"):
    """Extract descriptors from two binary images and return their Euclidean distance."""
    d1 = extract_descriptor(binary1, method=method)
    d2 = extract_descriptor(binary2, method=method)
    return euclidean_distance(d1, d2)
 
 
def pairwise_distance_matrix(shape_dict, method="similitude"):
    """
    Compute a pairwise Euclidean distance matrix for a dict of {name: binary_image}.
    Returns (names, matrix).
    """
    names = list(shape_dict.keys())
    n = len(names)
    descriptors = {name: extract_descriptor(img, method=method)
                   for name, img in shape_dict.items()}
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = euclidean_distance(descriptors[names[i]], descriptors[names[j]])
    return names, matrix
 
 
# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

def classify_pairs(shape_dict, threshold, method="similitude"):
    """
    For every pair of shapes, predict match/no-match and return results.
    Returns a list of dicts with keys: shape_a, shape_b, distance, predicted_match.
    """
    names = list(shape_dict.keys())
    descriptors = {n: extract_descriptor(img, method=method) for n, img in shape_dict.items()}
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            dist = euclidean_distance(descriptors[a], descriptors[b])
            results.append({
                "shape_a": a,
                "shape_b": b,
                "distance": dist,
                "predicted_match": dist <= threshold,
            })
    return results
 
 
# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────
 
def _run(binary, transform_fn, param_values, param_name, shape_name, method):
    """Generic helper: apply a transform at each param value, record distance."""
    records = []
    for val in param_values:
        transformed = largest_component(transform_fn(binary, val))
        dist = descriptor_distance(binary, transformed, method=method)
        records.append({
            "shape": shape_name,
            "distortion": param_name,
            "param": val,
            "distance": dist,
        })
    return records
 
 
def run_rotation(binary, shape_name, angles=None, method="similitude"):
    if angles is None:
        angles = list(range(0, 181, 15))
    return _run(binary, lambda b, a: rotate_image(b, a),
                angles, "rotation", shape_name, method)
 
 
def run_scaling(binary, shape_name, scales=None, method="similitude"):
    if scales is None:
        scales = [0.4, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75]
    return _run(binary, lambda b, s: scale_image(b, s),
                scales, "scaling", shape_name, method)
 
 
def run_translation(binary, shape_name, shifts=None, method="similitude"):
    if shifts is None:
        shifts = [(0, 0), (10, 10), (20, -15), (-25, 30), (40, -40), (60, 60)]
    records = []
    for shift_y, shift_x in shifts:
        transformed = largest_component(translate_image(binary, shift_y, shift_x))
        dist = descriptor_distance(binary, transformed, method=method)
        records.append({
            "shape": shape_name,
            "distortion": "translation",
            "param": (shift_y, shift_x),
            "distance": dist,
        })
    return records
 
 
def run_gaussian_noise(binary, shape_name, sigmas=None, method="similitude"):
    if sigmas is None:
        sigmas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    return _run(binary, lambda b, s: add_gaussian_noise(b, sigma=s),
                sigmas, "gaussian_noise", shape_name, method)
 
 
def run_salt_pepper(binary, shape_name, amounts=None, method="similitude"):
    if amounts is None:
        amounts = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    return _run(binary, lambda b, a: add_salt_pepper_noise(b, amount=a),
                amounts, "salt_pepper", shape_name, method)
 
 
def run_erosion(binary, shape_name, radii=None, method="similitude"):
    if radii is None:
        radii = [1, 2, 3, 5, 7, 10]
    return _run(binary, lambda b, r: erode_boundary(b, radius=r),
                radii, "erosion", shape_name, method)
 
 
def run_dilation(binary, shape_name, radii=None, method="similitude"):
    if radii is None:
        radii = [1, 2, 3, 5, 7, 10]
    return _run(binary, lambda b, r: dilate_boundary(b, radius=r),
                radii, "dilation", shape_name, method)
 
 
def run_blur(binary, shape_name, sigmas=None, method="similitude"):
    """Apply blur-then-threshold at each sigma and record descriptor distance."""
    if sigmas is None:
        sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    return _run(binary, lambda b, s: blur_and_threshold(b, sigma=s),
                sigmas, "blur", shape_name, method)
 
 
 
# ─────────────────────────────────────────────
# Accuracy analysis
# ─────────────────────────────────────────────
 
def collect_same_diff_distances(shape_dict, method="similitude"):
    """
    Build same-class distances (original vs transformed variants)
    and different-class distances (cross-shape pairs).
    """
    names, dist_matrix = pairwise_distance_matrix(shape_dict, method=method)
    n = len(names)
 
    same_distances = []
    test_angles  = [15, 30, 45, 90, 135]
    test_sigmas  = [0.03, 0.05, 0.08]
    test_amounts = [0.03, 0.05, 0.08]
 
    for name, img in shape_dict.items():
        d_orig = extract_descriptor(img, method=method)
        for angle in test_angles:
            t = largest_component(rotate_image(img, angle))
            same_distances.append(euclidean_distance(d_orig, extract_descriptor(t, method=method)))
        for sig in test_sigmas:
            t = largest_component(add_gaussian_noise(img, sigma=sig))
            same_distances.append(euclidean_distance(d_orig, extract_descriptor(t, method=method)))
        for amt in test_amounts:
            t = largest_component(add_salt_pepper_noise(img, amount=amt))
            same_distances.append(euclidean_distance(d_orig, extract_descriptor(t, method=method)))
 
    diff_distances = [dist_matrix[i, j]
                      for i in range(n) for j in range(n) if i != j]
    return same_distances, diff_distances
 
 
def per_distortion_accuracy(shape_dict, diff_distances, method="similitude"):
    """Compute accuracy for each individual distortion type."""
    distortions = [
        ("Rotation 15°",    lambda i: largest_component(rotate_image(i, 15))),
        ("Rotation 90°",    lambda i: largest_component(rotate_image(i, 90))),
        ("Gaussian σ=0.05", lambda i: largest_component(add_gaussian_noise(i, 0.05))),
        ("S&P amt=0.05",    lambda i: largest_component(add_salt_pepper_noise(i, 0.05))),
        ("Erosion r=3",     lambda i: largest_component(erode_boundary(i, 3))),
        ("Dilation r=3",    lambda i: largest_component(dilate_boundary(i, 3))),
    ]
    results = []
    for dist_name, fn in distortions:
        s_dists = []
        for name, img in shape_dict.items():
            d_orig = extract_descriptor(img, method=method)
            t = fn(img)
            s_dists.append(euclidean_distance(d_orig, extract_descriptor(t, method=method)))
        st = compute_matching_stats(s_dists, diff_distances)
        results.append({"distortion": dist_name, **st})
    return results

def preprocessing_comparison(image, shape_name, method="similitude"):
    """
    Compare descriptor stability with and without full preprocessing.
    Returns the intermediate images and the descriptor distance
    between the raw-binary version and the fully preprocessed version.
    """
    from preprocessing import to_grayscale, binarize_image, clean_binary, largest_component

    gray = to_grayscale(image)
    binary = binarize_image(gray)

    # Without cleaning: just take largest component from raw binary
    raw_component = largest_component(binary)

    # With full preprocessing
    cleaned = clean_binary(binary)
    clean_component = largest_component(cleaned)

    d_raw = extract_descriptor(raw_component, method=method)
    d_clean = extract_descriptor(clean_component, method=method)
    dist = euclidean_distance(d_raw, d_clean)

    return {
        "shape": shape_name,
        "gray": gray,
        "binary": binary,
        "raw_component": raw_component,
        "cleaned": cleaned,
        "clean_component": clean_component,
        "distance": dist,
    }

def print_summary_table(stats, per_dist, save_path=None):
    """Print the overall stats table and per-distortion accuracy table."""
    print("\n  Overall Matching Statistics:")
    print(f"  {'Metric':<35} {'Value':>10}")
    print("  " + "-"*47)
    print(f"  {'Same-shape mean distance':<35} {stats['same_mean']:>10.4f}")
    print(f"  {'Same-shape std':<35} {stats['same_std']:>10.4f}")
    print(f"  {'Different-shape mean distance':<35} {stats['diff_mean']:>10.4f}")
    print(f"  {'Different-shape std':<35} {stats['diff_std']:>10.4f}")
    print(f"  {'Classification threshold':<35} {stats['threshold']:>10.4f}")
    print(f"  {'Overall accuracy':<35} {stats['accuracy']*100:>9.1f}%")

    print("\n  Per-Distortion Accuracy:")
    print(f"  {'Distortion':<25} {'Same μ':>8} {'Same σ':>8} {'Threshold':>10} {'Accuracy':>10}")
    print("  " + "-"*65)
    for row in per_dist:
        print(f"  {row['distortion']:<25} {row['same_mean']:>8.4f} "
              f"{row['same_std']:>8.4f} {row['threshold']:>10.4f} "
              f"{row['accuracy']*100:>9.1f}%")

    if save_path is not None:
        import csv

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["Overall Matching Statistics"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Same-shape mean distance", stats["same_mean"]])
            writer.writerow(["Same-shape std", stats["same_std"]])
            writer.writerow(["Different-shape mean distance", stats["diff_mean"]])
            writer.writerow(["Different-shape std", stats["diff_std"]])
            writer.writerow(["Classification threshold", stats["threshold"]])
            writer.writerow(["Overall accuracy", stats["accuracy"]])

            writer.writerow([])
            writer.writerow(["Per-Distortion Accuracy"])
            writer.writerow(["Distortion", "Same Mean", "Same Std", "Threshold", "Accuracy"])

            for row in per_dist:
                writer.writerow([
                    row["distortion"],
                    row["same_mean"],
                    row["same_std"],
                    row["threshold"],
                    row["accuracy"],
                ])

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
 
def plot_pipeline(stages, titles, suptitle="Preprocessing Pipeline", save_path=None):
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, img, title in zip(axes, stages, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()

def plot_preprocessing_comparison(result, save_path=None):
    """
    Show how preprocessing changes the extracted shape mask.
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))

    images = [
        result["gray"],
        result["binary"],
        result["raw_component"],
        result["cleaned"],
        result["clean_component"],
    ]
    titles = [
        "Grayscale",
        "Raw Binary",
        "Largest Component\n(no cleaning)",
        "Cleaned Binary",
        "Largest Component\n(with cleaning)",
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(
        f"Preprocessing Comparison — {result['shape']}  |  Descriptor distance = {result['distance']:.4f}",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        plt.close()
    else:
        plt.show()
        
def plot_shape_gallery(shape_dict, title="Shape Gallery", save_path=None):
    names = list(shape_dict.keys())
    n = len(names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax, name in zip(axes.flat, names):
        ax.imshow(shape_dict[name], cmap='gray')
        ax.set_title(name, fontsize=12)
        ax.axis('off')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
def plot_experiment(x_vals, y_dict, baseline, xlabel, title, save_path=None, x_labels=None):
    """
    Plot same-shape distortion curves for two shapes plus a cross-shape baseline.
    y_dict: {label: [distances]}
    baseline: float — cross-shape separability reference
    """
    colors  = ['steelblue', 'tomato']
    markers = ['o', 's']
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, y) in enumerate(y_dict.items()):
        ax.plot(x_vals, y, marker=markers[i], color=colors[i],
                linewidth=2, label=f"{label} (same-shape)")
    ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5,
               label=f"Cross-shape baseline ({baseline:.4f})")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Euclidean Distance", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
 
 
def plot_combined_noise(noise_levels, gn_dict, sp_dict, gn_baseline, sp_baseline,
                        save_path=None):
    """
    Combined noise graph showing Gaussian and salt-and-pepper on one plot.
    gn_dict: {label: [distances]} for Gaussian noise shapes
    sp_dict: {label: [distances]} for salt-and-pepper noise shapes
    Both share the same x-axis (noise level 0→0.20).
    """
    colors  = ['steelblue', 'royalblue', 'tomato', 'firebrick']
    markers = ['o', '^', 's', 'D']
    styles  = ['-', '--', '-', '--']
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    all_lines = (
        [(label, y, colors[i], markers[i], styles[i], "Gaussian")
         for i, (label, y) in enumerate(gn_dict.items())] +
        [(label, y, colors[2+i], markers[2+i], styles[2+i], "S&P")
         for i, (label, y) in enumerate(sp_dict.items())]
    )
 
    for label, y, color, marker, style, noise_type in all_lines:
        ax.plot(noise_levels, y, marker=marker, color=color, linestyle=style,
                linewidth=2, label=f"{label} [{noise_type}]")
 
    # Draw both baselines
    ax.axhline(gn_baseline, color='steelblue', linestyle=':', linewidth=1.5,
               label=f"Gaussian cross-shape baseline ({gn_baseline:.4f})")
    ax.axhline(sp_baseline, color='tomato', linestyle=':', linewidth=1.5,
               label=f"S&P cross-shape baseline ({sp_baseline:.4f})")
 
    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Euclidean Distance", fontsize=12)
    ax.set_title("Phase 3 – Distance vs Noise Level (Gaussian vs Salt-and-Pepper)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
 
def plot_segmentation_error(radii, erosion_dict, dilation_dict, baseline,
                             save_path=None):
    """
    Dedicated imperfect segmentation graph.
    Shows erosion (under-segmentation) vs dilation (over-segmentation)
    for two shapes on one plot.
    """
    colors  = ['steelblue', 'royalblue', 'tomato', 'firebrick']
    markers = ['o', '^', 's', 'D']
 
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for i, (label, y) in enumerate(erosion_dict.items()):
        ax.plot(radii, y, marker=markers[i], color=colors[i], linestyle='-',
                linewidth=2, label=f"{label} [erosion / under-seg]")
 
    for i, (label, y) in enumerate(dilation_dict.items()):
        ax.plot(radii, y, marker=markers[2+i], color=colors[2+i], linestyle='--',
                linewidth=2, label=f"{label} [dilation / over-seg]")
 
    ax.axhline(baseline, color='black', linestyle=':', linewidth=1.5,
               label=f"Cross-shape baseline ({baseline:.4f})")
 
    ax.set_xlabel("Morphological Radius (px)", fontsize=12)
    ax.set_ylabel("Euclidean Distance", fontsize=12)
    ax.set_title("Phase 3 – Imperfect Segmentation: Erosion vs Dilation",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
def plot_histogram(same_distances, diff_distances, title="Distance Histogram",
                   threshold=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(same_distances, bins=20, alpha=0.7, color='steelblue', label='Same shape')
    ax.hist(diff_distances, bins=20, alpha=0.7, color='tomato', label='Different shape')
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--',
                   linewidth=1.5, label=f'Threshold = {threshold:.4f}')
    ax.set_xlabel("Euclidean Distance", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
def plot_distance_matrix(names, matrix, title="Pairwise Distance Matrix", save_path=None):
    n = len(names)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    plt.colorbar(im, ax=ax, label='Euclidean Distance')
    mid = matrix.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                    color='white' if matrix[i, j] > mid else 'black', fontsize=7)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
def plot_match_vs_mismatch(img_a1, img_b1, dist1, label_a1, label_b1,
                            img_a2, img_b2, dist2, label_a2, label_b2,
                            save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(img_a1, cmap='gray'); axes[0, 0].set_title(label_a1); axes[0, 0].axis('off')
    axes[0, 1].imshow(img_b1, cmap='gray'); axes[0, 1].set_title(label_b1); axes[0, 1].axis('off')
    axes[1, 0].imshow(img_a2, cmap='gray'); axes[1, 0].set_title(label_a2); axes[1, 0].axis('off')
    axes[1, 1].imshow(img_b2, cmap='gray'); axes[1, 1].set_title(label_b2); axes[1, 1].axis('off')
    fig.text(0.5, 0.96, f"MATCH — distance: {dist1:.4f}",
             ha='center', fontsize=11, color='green', fontweight='bold')
    fig.text(0.5, 0.50, f"MISMATCH — distance: {dist2:.4f}",
             ha='center', fontsize=11, color='red', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show()
 
 
# ─────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────
 
def find_hard_easy_pairs(names, matrix):
    """Return pairs sorted hardest (smallest distance) to easiest (largest)."""
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], matrix[i, j]))
    pairs.sort(key=lambda x: x[2])
    return pairs
 
 
def print_failure_report(shape_dict, method="similitude"):
    """Full failure case analysis and written discussion."""
    names, matrix = pairwise_distance_matrix(shape_dict, method=method)
    same_dists, diff_dists = collect_same_diff_distances(shape_dict, method=method)
    per_dist = per_distortion_accuracy(shape_dict, diff_dists, method=method)
    pairs = find_hard_easy_pairs(names, matrix)
    failures = [row["distortion"] for row in per_dist if row["same_mean"] > 0.02]
 
    print("\n" + "="*60)
    print("  FAILURE CASE ANALYSIS")
    print("="*60)
 
    print("\n  Pairs ranked hardest → easiest to separate:")
    print(f"  {'Pair':<32} {'Distance':>10}  {'Difficulty'}")
    print("  " + "-"*55)
    for i, (a, b, d) in enumerate(pairs):
        difficulty = "HARD" if i < 3 else ("EASY" if i >= len(pairs) - 3 else "    ")
        print(f"  {a+' vs '+b:<32} {d:>10.4f}  {difficulty}")
 
    print("\n  Distortions causing most descriptor drift:")
    if failures:
        for f in failures:
            print(f"    ✗  {f} — same-shape distance exceeded stability threshold")
    else:
        print("    ✓  All distortions within acceptable range")
 
    print("\n  Per-distortion accuracy summary:")
    for row in per_dist:
        status = "✓" if row["accuracy"] >= 0.75 else "✗"
        print(f"    {status}  {row['distortion']:<22}  accuracy = {row['accuracy']*100:.1f}%")
 
    hardest = pairs[0]
    easiest = pairs[-1]