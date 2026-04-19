# Shape Matching and Similarity Detection Using Moment-Based Descriptors

## Overview
This project implements a shape matching system that identifies and compares objects
based on their geometric structure, then experimentally tests how robust those
descriptors are under real-world image degradation and geometric transformations.

Given input images, the system segments objects, extracts shape descriptors using
similitude moments (Hu's 7 rotation-invariant moments derived explicitly from
normalized central moments), and computes similarity between shapes using Euclidean
distance. The core of the project is an experiment-driven analysis that pushes the
descriptors until they fail — finding exactly where and why moment-based matching
breaks down.

## Central Question
How robust are moment-based shape descriptors for shape matching under geometric
transformations and image degradation?

## Project Structure
Shape-Matching-and-Similarity-Detection/
├── main.py            # Entry point — runs all phases
├── descriptors.py     # Similitude and Hu moment feature extraction
├── preprocessing.py   # Grayscale, binarization, morphological cleaning
├── transforms.py      # Rotation, scaling, translation, noise, erosion, dilation
├── matching.py        # Euclidean distance, threshold, accuracy metrics
├── experiments.py     # Experiment runner, plotting, failure analysis
├── silhouettes/       # Real object silhouette images (PNG/JPG)
└── outputs/           # All generated figures and tables

## Pipeline
Each image goes through the following stages:
1. Load image
2. Convert to grayscale
3. Threshold to binary (Otsu, border-aware polarity detection)
4. Clean with morphological operations (opening, closing, small object removal)
5. Extract largest connected component
6. Compute 7-dimensional similitude moment descriptor
7. Compare descriptors with Euclidean distance

## Phases

### Phase 1 — Preprocessing Pipeline Demo
Demonstrates the full preprocessing pipeline on a noisy triangle, showing each
intermediate stage from raw input to final binary mask.

### Phase 2 — Shape Gallery
Displays all 6 synthetic base shapes: Circle, Ellipse, Rectangle, Triangle,
Pentagon, and Star. Also shows the preprocessing comparison (raw vs cleaned binary)
and the real silhouette gallery with pipeline demo.

### Phase 3 — Experiments
Runs controlled distortion experiments across all shapes and records descriptor
distance vs distortion intensity. Distortions tested:
- Rotation (0 to 180 degrees)
- Scaling (0.4x to 1.75x)
- Translation (up to 60px offset)
- Gaussian noise (sigma = 0.01 to 0.20)
- Salt-and-pepper noise (amount = 0.01 to 0.20)
- Boundary erosion (radius 1 to 15px)
- Blur before thresholding (sigma = 1.0 to 5.0px)
- Imperfect segmentation (erosion vs dilation)
- Synthetic vs real silhouette comparison under rotation and S&P noise

### Phase 4 — Failure Analysis and Accuracy Report
- Pairwise distance matrix across all synthetic shapes
- Combined synthetic + real silhouette distance matrix
- Same-shape vs different-shape distance histogram
- Match vs mismatch visualization
- Per-distortion accuracy table
- Ranked failure case analysis identifying hardest and easiest pairs to separate

## Key Findings
- Rotation, scale, and translation invariance hold across all tested shapes and
  intensities — same-shape distances remain near zero well below the cross-shape
  baseline.
- Salt-and-pepper noise is the primary failure mode — descriptor distance crosses
  the cross-shape baseline at approximately 12% noise, causing matching to fail.
- Gaussian noise is well tolerated — descriptors remain stable up to sigma = 0.15
  before meaningful drift occurs.
- Circle vs Pentagon and Triangle vs Star are the hardest pairs to separate —
  both pairs fall below the global classification threshold (0.0105), meaning moment
  descriptors cannot reliably distinguish them. This is a fundamental limitation:
  Hu moments capture mass distribution, not boundary shape, so a smoothed pentagon
  and a circle appear nearly identical in moment space.
- Real silhouettes are more sensitive to S&P noise than synthetic shapes — the
  cat silhouette drifts faster than the triangle under equivalent noise levels due
  to its more complex boundary.
- Overall classification accuracy is 91.7% with a midpoint threshold. The accuracy
  ceiling of 77.8% per distortion is caused by the two intrinsically hard pairs
  remaining below the threshold regardless of distortion type.

## Requirements
numpy
matplotlib
scikit-image
scipy

Install with:
pip install numpy matplotlib scikit-image scipy

## Usage
python3 main.py

All outputs are saved to the outputs/ directory. To use real silhouettes, place
image files (PNG, JPG, WEBP, BMP) in the silhouettes/ folder before running.

## Adding Real Silhouettes
Place any image file in the silhouettes/ folder. The system automatically:
- Detects background polarity (works with both black-on-white and white-on-black)
- Applies the full preprocessing pipeline
- Includes the silhouette in the distance matrix and comparison experiments

## Descriptor Method
The default descriptor is similitude — Hu's 7 rotation-invariant moments derived
explicitly from normalized central moments up to order 3. A hu method is also
available using scikit-image's built-in implementation. Both are compared against
the same Euclidean distance matching framework.