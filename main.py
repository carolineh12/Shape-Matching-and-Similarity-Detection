import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from skimage.draw import disk, polygon, rectangle, ellipse
from skimage import io as skio
 
# Team Member 1
from preprocessing import (
    to_grayscale, binarize_image, clean_binary, largest_component
)
from transforms import rotate_image, add_gaussian_noise, add_salt_pepper_noise
from descriptors import extract_descriptor
 
# Team Member 2
from matching import euclidean_distance, compute_matching_stats
from experiments import (
    # similarity
    descriptor_distance, pairwise_distance_matrix,
    # classifier
    is_match, classify_pairs,
    # experiment runner
    run_rotation, run_scaling, run_translation,
    run_gaussian_noise, run_salt_pepper, run_erosion, run_dilation, run_blur,
    # accuracy analysis
    collect_same_diff_distances, per_distortion_accuracy, print_summary_table,
    # plotting
    plot_pipeline, plot_shape_gallery, plot_experiment,
    plot_combined_noise, plot_segmentation_error,
    plot_histogram, plot_distance_matrix, plot_match_vs_mismatch,
    # failure analysis
    print_failure_report,
)
 
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
METHOD = "similitude"
 
 
# ─────────────────────────────────────────────
# Shape generators  (Team Member 1)
# ─────────────────────────────────────────────
 
def _blank(size=256):
    return np.zeros((size, size), dtype=np.uint8)
 
def make_circle(size=256, radius=60):
    img = _blank(size); rr, cc = disk((size//2, size//2), radius, shape=img.shape); img[rr, cc] = 1; return img
 
def make_ellipse(size=256, r_radius=40, c_radius=70):
    img = _blank(size); rr, cc = ellipse(size//2, size//2, r_radius, c_radius, shape=img.shape); img[rr, cc] = 1; return img
 
def make_rectangle(size=256, height=80, width=140):
    img = _blank(size)
    start = ((size - height) // 2, (size - width) // 2)
    rr, cc = rectangle(start=start, extent=(height, width), shape=img.shape); img[rr, cc] = 1; return img
 
def make_triangle(size=256):
    img = _blank(size)
    rr, cc = polygon([50, 200, 200], [128, 55, 201], img.shape); img[rr, cc] = 1; return img
 
def make_pentagon(size=256, radius=80):
    img = _blank(size)
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)[:-1]
    r = size//2 + radius * np.sin(angles)
    c = size//2 + radius * np.cos(angles)
    rr, cc = polygon(r.astype(int), c.astype(int), img.shape); img[rr, cc] = 1; return img
 
def make_star(size=256, outer=80, inner=35, points=5):
    img = _blank(size); cx, cy = size//2, size//2
    ao = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi, points, endpoint=False)
    ai = ao + np.pi / points
    r, c = [], []
    for a_o, a_i in zip(ao, ai):
        r += [cx + outer*np.sin(a_o), cx + inner*np.sin(a_i)]
        c += [cy + outer*np.cos(a_o), cy + inner*np.cos(a_i)]
    rr, cc = polygon(np.array(r), np.array(c), img.shape); img[rr, cc] = 1; return img
 
 
SHAPES = {
    "Circle":    make_circle(),
    "Ellipse":   make_ellipse(),
    "Rectangle": make_rectangle(),
    "Triangle":  make_triangle(),
    "Pentagon":  make_pentagon(),
    "Star":      make_star(),
}
 
 
# ─────────────────────────────────────────────
# Real silhouette loader  (Team Member 1)
# ─────────────────────────────────────────────
 
SILHOUETTE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silhouettes")
 
def load_silhouettes():
    """
    Load any image files found in the silhouettes/ folder and run them through
    the full preprocessing pipeline to produce clean binary masks.
    Supports .png, .jpg, .jpeg, .webp — any filename is accepted.
    Returns a dict {name: binary_image}.
    """
    from preprocessing import preprocess_image
    supported = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    result = {}
    if not os.path.isdir(SILHOUETTE_DIR):
        print(f"  Warning: silhouettes/ folder not found at {SILHOUETTE_DIR}")
        return result
    for fname in sorted(os.listdir(SILHOUETTE_DIR)):
        if not fname.lower().endswith(supported):
            continue
        path = os.path.join(SILHOUETTE_DIR, fname)
        # Use filename without extension as the display name
        name = os.path.splitext(fname)[0].replace('_', ' ').replace('-', ' ').title()
        try:
            raw = skio.imread(path)
            # Handle RGBA, palette, or grayscale images — convert to RGB first
            if raw.ndim == 2:
                raw = np.stack([raw]*3, axis=-1)
            elif raw.shape[-1] == 4:
                raw = raw[..., :3]  # drop alpha channel
            _, _, _, component = preprocess_image(raw)
            if component.sum() == 0:
                print(f"  Warning: empty mask for {fname}, skipping")
                continue
            result[name] = component
            print(f"  Loaded silhouette: {name}")
        except Exception as e:
            print(f"  Warning: could not load {fname} — {e}")
    return result
 
SILHOUETTES = load_silhouettes()
 
 
def _cross_shape_distance(shape_a, shape_b):
    """Distance between two different shapes — used as separability baseline on graphs."""
    return euclidean_distance(
        extract_descriptor(shape_a, method=METHOD),
        extract_descriptor(shape_b, method=METHOD),
    )
 
 
# ─────────────────────────────────────────────
# Phase 1 – Pipeline demo  (Team Member 1)
# ─────────────────────────────────────────────
 
def phase1():
    print("\n" + "="*60)
    print("PHASE 1: Preprocessing Pipeline Demo")
    print("="*60)
 
    triangle = SHAPES["Triangle"]
    noisy    = add_gaussian_noise(triangle, sigma=0.08)
    gray     = to_grayscale(noisy.astype(float))
    binary   = binarize_image(gray)
    cleaned  = clean_binary(binary)
    comp     = largest_component(cleaned)
 
    plot_pipeline(
        stages=[noisy, gray, binary, cleaned, comp],
        titles=["Original (noisy)", "Grayscale", "Binary (Otsu)", "Cleaned", "Largest Component"],
        suptitle="Phase 1 – Full Preprocessing Pipeline",
        save_path=os.path.join(OUTPUT_DIR, "phase1_pipeline.png"),
    )
    print("  Saved: phase1_pipeline.png")
    desc = extract_descriptor(comp, method=METHOD)
    print(f"  Descriptor (Triangle, 7-dim): {np.round(desc, 4)}")
 
 
# ─────────────────────────────────────────────
# Phase 2 – Shape gallery  (Team Member 1)
# ─────────────────────────────────────────────
 
def phase2():
    print("\n" + "="*60)
    print("PHASE 2: Base Shape Gallery")
    print("="*60)
 
    plot_shape_gallery(
        SHAPES,
        title="Phase 2 – Shape Gallery (6 shapes)",
        save_path=os.path.join(OUTPUT_DIR, "phase2_shapes.png"),
    )
    print("  Saved: phase2_shapes.png")
 
 
# ─────────────────────────────────────────────
# Phase 3 – Experiments  (Team Member 2)
# ─────────────────────────────────────────────
 
 
def phase2b():
    print("\n" + "="*60)
    print("PHASE 2b: Real Silhouette Gallery")
    print("="*60)
 
    if not SILHOUETTES:
        print("  No silhouettes found — place image files in the silhouettes/ folder.")
        print(f"  Expected folder: {SILHOUETTE_DIR}")
        return
 
    plot_shape_gallery(
        SILHOUETTES,
        title="Phase 2b – Real Object Silhouettes",
        save_path=os.path.join(OUTPUT_DIR, "phase2_silhouettes.png"),
    )
    print("  Saved: phase2_silhouettes.png")
 
    # Show pipeline on one silhouette to demonstrate real-image preprocessing
    from preprocessing import to_grayscale, binarize_image, clean_binary, largest_component
    import os as _os
    # Use the first available silhouette for the pipeline demo
    supported = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    first_file = next((f for f in sorted(_os.listdir(SILHOUETTE_DIR))
                       if f.lower().endswith(supported)), None)
    if first_file is None:
        print("  No silhouettes found for pipeline demo, skipping.")
        return
    first_name = _os.path.splitext(first_file)[0].replace('_',' ').title()
    raw = skio.imread(_os.path.join(SILHOUETTE_DIR, first_file))
    if raw.ndim == 2:
        raw = np.stack([raw]*3, axis=-1)
    elif raw.shape[-1] == 4:
        raw = raw[..., :3]
    gray    = to_grayscale(raw)
    binary  = binarize_image(gray)
    cleaned = clean_binary(binary)
    comp    = largest_component(cleaned)
    plot_pipeline(
        stages=[raw, gray, binary, cleaned, comp],
        titles=["Raw silhouette", "Grayscale", "Binary (Otsu)", "Cleaned", "Largest Component"],
        suptitle=f"Phase 2b – Real Silhouette Preprocessing Pipeline ({first_name})",
        save_path=os.path.join(OUTPUT_DIR, "phase2_silhouette_pipeline.png"),
    )
    print("  Saved: phase2_silhouette_pipeline.png")
 
 
def phase3():
    print("\n" + "="*60)
    print("PHASE 3: Experiments")
    print("="*60)
 
    triangle  = SHAPES["Triangle"]
    rectangle = SHAPES["Rectangle"]
    circle    = SHAPES["Circle"]
    star      = SHAPES["Star"]
    pentagon  = SHAPES["Pentagon"]
 
    # --- Rotation: Triangle (asymmetric) vs Rectangle (symmetric) ---
    angles = list(range(0, 181, 15))
    rot_tri  = [r["distance"] for r in run_rotation(triangle,  "Triangle",  angles, METHOD)]
    rot_rect = [r["distance"] for r in run_rotation(rectangle, "Rectangle", angles, METHOD)]
    plot_experiment(angles,
        {"Triangle (asymmetric)": rot_tri, "Rectangle (symmetric)": rot_rect},
        _cross_shape_distance(triangle, rectangle),
        "Rotation Angle (degrees)", "Phase 3 – Distance vs Rotation Angle",
        save_path=os.path.join(OUTPUT_DIR, "phase3_rotation.png"))
    print("  Saved: phase3_rotation.png")
 
    # --- Scaling: Circle (simple) vs Star (complex boundary) ---
    scales = [0.4, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75]
    sc_circ = [r["distance"] for r in run_scaling(circle, "Circle", scales, METHOD)]
    sc_star = [r["distance"] for r in run_scaling(star,   "Star",   scales, METHOD)]
    plot_experiment(scales,
        {"Circle (simple)": sc_circ, "Star (complex)": sc_star},
        _cross_shape_distance(circle, star),
        "Scale Factor", "Phase 3 – Distance vs Scale Factor",
        save_path=os.path.join(OUTPUT_DIR, "phase3_scaling.png"))
    print("  Saved: phase3_scaling.png")
 
    # --- Translation: Circle vs Triangle (both should be perfectly invariant) ---
    shifts = [(0,0),(10,10),(20,-15),(-25,30),(40,-40),(60,60)]
    shift_labels = [f"({sy},{sx})" for sy, sx in shifts]
    tr_circ = [r["distance"] for r in run_translation(circle,   "Circle",   shifts, METHOD)]
    tr_tri  = [r["distance"] for r in run_translation(triangle, "Triangle", shifts, METHOD)]
    plot_experiment(list(range(len(shifts))),
        {"Circle": tr_circ, "Triangle": tr_tri},
        _cross_shape_distance(circle, triangle),
        "Translation (dy, dx)", "Phase 3 – Distance vs Translation",
        save_path=os.path.join(OUTPUT_DIR, "phase3_translation.png"),
        x_labels=shift_labels)
    print("  Saved: phase3_translation.png")
 
    # --- Gaussian noise: Triangle (sharp corners) vs Star (thin spikes) ---
    sigmas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    gn_tri  = [r["distance"] for r in run_gaussian_noise(triangle, "Triangle", sigmas, METHOD)]
    gn_star = [r["distance"] for r in run_gaussian_noise(star,     "Star",     sigmas, METHOD)]
    plot_experiment(sigmas,
        {"Triangle (sharp corners)": gn_tri, "Star (thin spikes)": gn_star},
        _cross_shape_distance(triangle, star),
        "Gaussian Noise Sigma", "Phase 3 – Distance vs Gaussian Noise",
        save_path=os.path.join(OUTPUT_DIR, "phase3_gaussian.png"))
    print("  Saved: phase3_gaussian.png")
 
    # --- Salt & pepper: Rectangle (clean edges) vs Star (complex boundary) ---
    amounts = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    sp_rect = [r["distance"] for r in run_salt_pepper(rectangle, "Rectangle", amounts, METHOD)]
    sp_star = [r["distance"] for r in run_salt_pepper(star,      "Star",      amounts, METHOD)]
    plot_experiment(amounts,
        {"Rectangle (clean edges)": sp_rect, "Star (complex boundary)": sp_star},
        _cross_shape_distance(rectangle, star),
        "Salt & Pepper Amount", "Phase 3 – Distance vs Salt-and-Pepper Noise",
        save_path=os.path.join(OUTPUT_DIR, "phase3_salt_pepper.png"))
    print("  Saved: phase3_salt_pepper.png")
 
    # --- Combined noise comparison: Gaussian vs Salt-and-Pepper on one plot ---
    noise_levels = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    plot_combined_noise(
        noise_levels,
        gn_dict={"Triangle (sharp corners)": gn_tri, "Star (thin spikes)": gn_star},
        sp_dict={"Rectangle (clean edges)": sp_rect, "Star (complex boundary)": sp_star},
        gn_baseline=_cross_shape_distance(triangle, star),
        sp_baseline=_cross_shape_distance(rectangle, star),
        save_path=os.path.join(OUTPUT_DIR, "phase3_noise_combined.png"),
    )
    print("  Saved: phase3_noise_combined.png")
 
    # --- Boundary erosion: Circle (smooth) vs Pentagon (multi-sided) ---
    radii = [1, 2, 3, 5, 7, 10]
    er_circ = [r["distance"] for r in run_erosion(circle,   "Circle",   radii, METHOD)]
    er_pent = [r["distance"] for r in run_erosion(pentagon, "Pentagon", radii, METHOD)]
    plot_experiment(radii,
        {"Circle (smooth boundary)": er_circ, "Pentagon (multi-sided)": er_pent},
        _cross_shape_distance(circle, pentagon),
        "Erosion Radius (px)", "Phase 3 – Distance vs Boundary Erosion",
        save_path=os.path.join(OUTPUT_DIR, "phase3_boundary_damage.png"))
    print("  Saved: phase3_boundary_damage.png")
 
    # --- Blur transform: Triangle vs Rectangle ---
    # Simulates slightly out-of-focus or low-resolution image before segmentation
    blur_sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    bl_tri  = [r["distance"] for r in run_blur(triangle,  "Triangle",  blur_sigmas, METHOD)]
    bl_rect = [r["distance"] for r in run_blur(rectangle, "Rectangle", blur_sigmas, METHOD)]
    plot_experiment(blur_sigmas,
        {"Triangle": bl_tri, "Rectangle": bl_rect},
        _cross_shape_distance(triangle, rectangle),
        "Blur Sigma (px)", "Phase 3 – Distance vs Blur (pre-threshold)",
        save_path=os.path.join(OUTPUT_DIR, "phase3_blur.png"))
    print("  Saved: phase3_blur.png")
 
    # --- Imperfect segmentation: erosion (under-seg) vs dilation (over-seg) ---
    # Triangle and Rectangle tested under both under- and over-segmentation
    radii = [1, 2, 3, 5, 7, 10]
    er_tri  = [r["distance"] for r in run_erosion( triangle,  "Triangle",  radii, METHOD)]
    di_tri  = [r["distance"] for r in run_dilation(triangle,  "Triangle",  radii, METHOD)]
    er_rect = [r["distance"] for r in run_erosion( rectangle, "Rectangle", radii, METHOD)]
    di_rect = [r["distance"] for r in run_dilation(rectangle, "Rectangle", radii, METHOD)]
    plot_segmentation_error(
        radii,
        erosion_dict={"Triangle": er_tri, "Rectangle": er_rect},
        dilation_dict={"Triangle": di_tri, "Rectangle": di_rect},
        baseline=_cross_shape_distance(triangle, rectangle),
        save_path=os.path.join(OUTPUT_DIR, "phase3_segmentation_error.png"),
    )
    print("  Saved: phase3_segmentation_error.png")
 
 
# ─────────────────────────────────────────────
# Phase 4 – Analysis  (Team Member 2)
# ─────────────────────────────────────────────
 
def phase4():
    print("\n" + "="*60)
    print("PHASE 4: Failure Analysis & Accuracy Report")
    print("="*60)
 
    names, matrix = pairwise_distance_matrix(SHAPES, method=METHOD)
    plot_distance_matrix(names, matrix,
        title="Phase 4 – Pairwise Descriptor Distance Matrix",
        save_path=os.path.join(OUTPUT_DIR, "phase4_distance_matrix.png"))
    print("  Saved: phase4_distance_matrix.png")
 
    # --- Silhouette vs synthetic distance matrix ---
    if SILHOUETTES:
        all_shapes = {**SHAPES, **SILHOUETTES}
        all_names, all_matrix = pairwise_distance_matrix(all_shapes, method=METHOD)
        plot_distance_matrix(all_names, all_matrix,
            title="Phase 4 – Synthetic + Silhouette Pairwise Distance Matrix",
            save_path=os.path.join(OUTPUT_DIR, "phase4_silhouette_distance_matrix.png"))
        print("  Saved: phase4_silhouette_distance_matrix.png")
    else:
        print("  Skipping silhouette distance matrix (no silhouettes loaded)")
 
 
    same_dists, diff_dists = collect_same_diff_distances(SHAPES, method=METHOD)
    stats    = compute_matching_stats(same_dists, diff_dists)
    per_dist = per_distortion_accuracy(SHAPES, diff_dists, method=METHOD)
 
    plot_histogram(same_dists, diff_dists,
        title="Phase 4 – Same-Shape vs Different-Shape Distances",
        threshold=stats["threshold"],
        save_path=os.path.join(OUTPUT_DIR, "phase4_histogram.png"))
    print("  Saved: phase4_histogram.png")
 
    circle    = SHAPES["Circle"]
    ellipse_s = SHAPES["Ellipse"]
    matched   = largest_component(rotate_image(circle, 45))
    d_match    = euclidean_distance(extract_descriptor(circle,    method=METHOD),
                                     extract_descriptor(matched,   method=METHOD))
    d_mismatch = euclidean_distance(extract_descriptor(circle,    method=METHOD),
                                     extract_descriptor(ellipse_s, method=METHOD))
    plot_match_vs_mismatch(
        circle, matched,   d_match,    "Circle (original)", "Circle (rotated 45°)",
        circle, ellipse_s, d_mismatch, "Circle",            "Ellipse",
        save_path=os.path.join(OUTPUT_DIR, "phase4_match_vs_mismatch.png"))
    print("  Saved: phase4_match_vs_mismatch.png")
 
    print_summary_table(stats, per_dist, save_path=os.path.join(OUTPUT_DIR, "phase4_results_table.csv"))
    print_failure_report(SHAPES, method=METHOD)
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
def main():
    print("\n" + "="*60)
    print("  Shape Matching – Moment-Based Descriptors")
    print("="*60)
    phase1()
    phase2()
    phase2b()
    phase3()
    phase4()
    print("\n" + "="*60)
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print("="*60)
 
 
if __name__ == "__main__":
    main()