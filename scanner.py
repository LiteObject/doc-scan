"""
Document scanner using OpenCV for automatic document detection and perspective correction.

This module processes images to detect document boundaries, apply perspective transformation,
and save a scanned version of the document.
"""

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np


# -----------------------------
# Content quality heuristics
# -----------------------------
def _to_gray(img):
    """Ensure grayscale uint8 image."""
    if img is None or img.size == 0:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    if g.dtype != np.uint8:
        g_min, g_max = float(g.min()), float(g.max())
        if g_max > g_min:
            g = ((g - g_min) * (255.0 / (g_max - g_min))).astype(np.uint8)
        else:
            g = np.zeros_like(g, dtype=np.uint8)
    return g


def is_near_blank(img):
    """Return True if the image looks near-blank (very low variance/edges or extreme black/white)."""
    g = _to_gray(img)
    if g is None or g.size == 0:
        return True
    std = float(g.std())
    edges = cv2.Canny(g, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / (g.size + 1e-6)
    # For binary-like cases, check black pixel ratio
    black_ratio = float(np.count_nonzero(g < 200)) / (g.size + 1e-6)
    # Heuristics
    if std < 5 and edge_ratio < 0.005:
        return True
    if black_ratio < 0.005 or black_ratio > 0.98:
        return True
    return False


def content_score(img):
    """Score how contentful the image is. Higher is better.
    Combines edge density and contrast; penalizes extreme black/white.
    """
    g = _to_gray(img)
    if g is None or g.size == 0:
        return -1.0
    edges = cv2.Canny(g, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / (g.size + 1e-6)
    contrast = float(g.std()) / 255.0
    black_ratio = float(np.count_nonzero(g < 200)) / (g.size + 1e-6)
    # Penalize extreme black/white dominance
    penalty = 0.0
    if black_ratio < 0.01 or black_ratio > 0.9:
        penalty += 0.3
    # Penalize excessive edge density (often looks noisy/ugly)
    # Start penalizing after ~12% of pixels are edges, scale up quickly
    if edge_ratio > 0.12:
        penalty += min(0.6, (edge_ratio - 0.12) * 3.0)
    # Favor some edges and good contrast, but reduce weight on edges
    return (0.45 * edge_ratio + 0.55 * contrast) - penalty


def order_points(pts):
    """Order points in clockwise: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    point_sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(point_sum)]  # top-left
    rect[2] = pts[np.argmax(point_sum)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def enhance_image_quality(img):
    """Apply various enhancement techniques to improve image quality"""
    # 1. Noise reduction using Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(denoised)

    # 3. Unsharp masking for better sharpness
    gaussian = cv2.GaussianBlur(enhanced_contrast, (0, 0), 2.0)
    unsharp_mask = cv2.addWeighted(enhanced_contrast, 1.5, gaussian, -0.5, 0)

    return unsharp_mask


def advanced_threshold(img):
    """Apply multiple thresholding techniques and return all versions"""
    # Otsu's thresholding
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive thresholding - mean
    adaptive_mean_thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # Adaptive thresholding - Gaussian
    adaptive_gaussian_thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # Niblack-like local thresholding
    kernel_size = 15
    k = 0.2
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (
        kernel_size * kernel_size
    )
    mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
    sqr_mean = cv2.filter2D((img.astype(np.float32)) ** 2, -1, kernel)
    variance = sqr_mean - mean**2
    std_dev = np.sqrt(np.maximum(variance, 0))
    threshold_map = mean + k * std_dev
    niblack_thresh = (img.astype(np.float32) > threshold_map).astype(np.uint8) * 255

    return otsu_thresh, adaptive_mean_thresh, adaptive_gaussian_thresh, niblack_thresh


def create_high_quality_scan(img):
    """Create high-quality scan using enhanced image processing"""
    print(f"Input image shape: {img.shape}")
    print(f"Input image type: {img.dtype}")
    print(f"Input image range: {img.min()}-{img.max()}")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("Converted to grayscale")
    else:
        gray = img.copy()
        print("Already grayscale")

    print(f"Grayscale shape: {gray.shape}")
    print(f"Grayscale range: {gray.min()}-{gray.max()}")

    enhanced_img = enhance_image_quality(gray)
    print(f"Enhanced image range: {enhanced_img.min()}-{enhanced_img.max()}")

    # Get all threshold versions
    otsu_thresh, adaptive_mean_thresh, adaptive_gaussian_thresh, niblack_thresh = (
        advanced_threshold(enhanced_img)
    )

    # Debug thresholding results
    print(f"Otsu threshold range: {otsu_thresh.min()}-{otsu_thresh.max()}")
    print(
        f"Adaptive mean range: {adaptive_mean_thresh.min()}-{adaptive_mean_thresh.max()}"
    )
    print(
        f"Adaptive Gaussian range: {adaptive_gaussian_thresh.min()}-{adaptive_gaussian_thresh.max()}"
    )
    print(f"Niblack range: {niblack_thresh.min()}-{niblack_thresh.max()}")

    # Create final optimized version by combining techniques
    # Use adaptive Gaussian as base and refine with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_processed = cv2.morphologyEx(
        adaptive_gaussian_thresh, cv2.MORPH_CLOSE, kernel
    )
    final_processed = cv2.morphologyEx(final_processed, cv2.MORPH_OPEN, kernel)

    print(f"Final processed range: {final_processed.min()}-{final_processed.max()}")

    return (
        enhanced_img,
        otsu_thresh,
        adaptive_mean_thresh,
        adaptive_gaussian_thresh,
        niblack_thresh,
        final_processed,
    )


def create_output_directory(output_path=None):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path:
        # Use provided output path
        if os.path.isabs(output_path):
            # Absolute path provided
            output_dir = output_path
        else:
            # Relative path provided
            output_dir = os.path.abspath(output_path)

        # If the path doesn't end with a timestamped folder, add one
        if not os.path.basename(output_dir).startswith("scanned_output_"):
            output_dir = os.path.join(output_dir, f"scanned_output_{timestamp}")
    else:
        # Default: create timestamped folder in current directory
        output_dir = f"scanned_output_{timestamp}"

    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_all_quality_versions(
    output_dir,
    enhanced_img,
    otsu_thresh,
    adaptive_mean_thresh,
    adaptive_gaussian_thresh,
    niblack_thresh,
    final_processed,
    perspective_corrected,
):
    """Save all quality versions for comparison"""
    quality_dir = f"{output_dir}/quality_comparison"
    os.makedirs(quality_dir, exist_ok=True)

    # Save all versions with descriptive names
    versions = [
        (
            perspective_corrected,
            "00_original_perspective_corrected.jpg",
            "Original after perspective correction",
        ),
        (
            enhanced_img,
            "01_enhanced_grayscale.jpg",
            "Enhanced grayscale (denoised + CLAHE + sharpened)",
        ),
        (otsu_thresh, "02_otsu_threshold.jpg", "Otsu's automatic thresholding"),
        (adaptive_mean_thresh, "03_adaptive_mean.jpg", "Adaptive mean thresholding"),
        (
            adaptive_gaussian_thresh,
            "04_adaptive_gaussian.jpg",
            "Adaptive Gaussian thresholding",
        ),
        (niblack_thresh, "05_niblack_local.jpg", "Niblack-like local thresholding"),
        (
            final_processed,
            "06_combined_optimized.jpg",
            "Combined optimized (RECOMMENDED)",
        ),
    ]

    for img_data, filename, _ in versions:
        filepath = f"{quality_dir}/{filename}"

        # Validate image before saving
        if img_data is None:
            print(f"   ⚠ {filename} - Image is None!")
            continue
        if img_data.size == 0:
            print(f"   ⚠ {filename} - Image is empty!")
            continue

        # Check if image is completely blank (all black or all white)
        unique_values = len(np.unique(img_data))
        img_min, img_max = img_data.min(), img_data.max()

        if unique_values == 1:
            print(f"   ⚠ {filename} - Image has only one value ({img_min})")
        elif img_min == img_max:
            print(f"   ⚠ {filename} - Image has uniform value ({img_min})")
        else:
            print(
                f"   ✓ {filename} (range: {img_min}-{img_max}, unique values: {unique_values})"
            )

        cv2.imwrite(filepath, img_data)

    # Create selection guide
    guide_content = """QUALITY COMPARISON GUIDE
========================

This folder contains 7 different processing versions of your scanned document.
Each version uses different techniques to optimize readability and quality.

FILE DESCRIPTIONS:
------------------
00_original_perspective_corrected.jpg - The document after perspective correction only
01_enhanced_grayscale.jpg - Enhanced version with noise reduction and contrast improvement
02_otsu_threshold.jpg - Black & white using Otsu's automatic threshold detection
03_adaptive_mean.jpg - Black & white using adaptive mean thresholding
04_adaptive_gaussian.jpg - Black & white using adaptive Gaussian thresholding  
05_niblack_local.jpg - Black & white using Niblack-like local thresholding
06_combined_optimized.jpg - **RECOMMENDED** Combined techniques with morphological cleanup

CHOOSING THE BEST VERSION:
--------------------------
✓ For most documents: Use 06_combined_optimized.jpg (RECOMMENDED)
✓ For documents with varying lighting: Try 04_adaptive_gaussian.jpg or 05_niblack_local.jpg
✓ For clean, high-contrast documents: Use 02_otsu_threshold.jpg
✓ For documents with handwriting: Try 01_enhanced_grayscale.jpg or 03_adaptive_mean.jpg
✓ For archival/color preservation: Use 00_original_perspective_corrected.jpg

The main folder also contains:
• RECOMMENDED_scanned_document.jpg (copy of 06_combined_optimized.jpg)
• GRAYSCALE_enhanced.jpg (copy of 01_enhanced_grayscale.jpg)
"""

    with open(f"{quality_dir}/README.txt", "w", encoding="utf-8") as f:
        f.write(guide_content)

    print("   ✓ README.txt (selection guide)")


def four_point_transform(image, pts):
    """Apply perspective transformation to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points for the transform
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # Apply perspective transform
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    output_warped = cv2.warpPerspective(
        image, transform_matrix, (max_width, max_height)
    )

    return output_warped


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document scanner that automatically detects and enhances scanned documents",
        epilog="Example: python scanner.py input_image.jpg --output ./scans/",
    )
    parser.add_argument("input_file", help="Path to the input image file to scan")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save intermediate processing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory path (default: creates timestamped folder in current directory)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum contour area (in resized-pixels) to accept as document (default: 1000)",
    )
    parser.add_argument(
        "--fallback-min-area",
        type=int,
        default=500,
        help="Minimum area to allow fallback quad if no primary match is found (default: 500)",
    )
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=0.04,
        help="If selected quad covers less than this fraction of the resized image, use full-frame fallback (default: 0.04 = 4%)",
    )
    parser.add_argument(
        "--prefer",
        type=str,
        choices=[
            "auto",
            "grayscale",
            "original",
            "otsu",
            "adaptive-mean",
            "adaptive-gaussian",
            "niblack",
            "combined",
        ],
        default="combined",
        help=(
            "Choose which variant to save as RECOMMENDED: "
            "auto (score-based), grayscale, original, otsu, adaptive-mean, adaptive-gaussian, niblack, combined"
        ),
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=["auto", "table", "text"],
        default="auto",
        help=(
            "Bias auto selection for certain content: 'table' favors clean binary and structured edges; "
            "'text' favors smoother grayscale/less edge-heavy variants; 'auto' leaves default behavior."
        ),
    )
    return parser.parse_args()


def main():
    """Main function to handle CLI arguments and run the scanner."""
    args = parse_arguments()

    input_file = args.input_file
    debug_mode = args.debug
    output_path = args.output
    min_area_arg = int(getattr(args, "min_area", 1000))
    fallback_min_area_arg = int(getattr(args, "fallback_min_area", 500))
    min_area_frac_arg = float(getattr(args, "min_area_frac", 0.04))
    prefer_variant = getattr(args, "prefer", "auto")
    doc_type_profile = getattr(args, "doc_type", "auto")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)

    # Load image
    image = cv2.imread(input_file)
    if image is None:
        print(
            f"Error: Unable to load image from '{input_file}'. "
            "Please check if it's a valid image file."
        )
        sys.exit(1)

    # Validate image dimensions
    if image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Invalid image dimensions!")
        sys.exit(1)

    print(f"Processing image: {input_file}")
    if output_path:
        print(f"Output will be saved to: {output_path}")
    if debug_mode:
        print("Debug mode enabled - intermediate images will be saved")

    orig = image.copy()

    # Safely calculate ratio to prevent division by zero
    if image.shape[0] == 0:
        print("Error: Image height is zero!")
        sys.exit(1)

    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)  # initial edges
    # Collect alternative edge images for debug saving later
    debug_alt_images = []
    # We'll save all debug images later into a single debug_processing folder
    if debug_mode:
        print(
            "Debug: Captured intermediate processing images (will save after processing)"
        )

    # Find contours
    try:
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
    except cv2.error as e:
        print(f"Error finding contours: {e}")
        sys.exit(1)

    if len(contours) == 0:
        print("Error: No contours found in the image!")
        sys.exit(1)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Debug: Print contour information
    if debug_mode:
        print(f"Debug: Found {len(contours)} contours")
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            print(
                f"  Contour {i}: area={area:.0f}, perimeter={peri:.0f}, vertices={len(approx)}"
            )

    # Adaptive detection settings
    screen_contour = None
    epsilon_candidates = [0.02, 0.03, 0.015, 0.025, 0.01, 0.035, 0.04, 0.045, 0.05]
    min_area = min_area_arg  # initial minimum area (configurable)

    # Track the largest valid quadrilateral (for fallback)
    best_fallback = {
        "contour": None,
        "area": 0,
        "source": "initial",
        "epsilon": None,
        "params": None,
    }

    for c in contours:
        area = cv2.contourArea(c)

        # Try to approximate to quadrilateral even if area is small, to consider as fallback
        peri = cv2.arcLength(c, True)
        local_quad = None
        local_eps = None
        for epsilon_factor in epsilon_candidates:
            approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)
            if len(approx) == 4:
                local_quad = approx
                local_eps = epsilon_factor
                break

        # Update best fallback if this quad is the largest seen so far
        if local_quad is not None and area > best_fallback["area"]:
            best_fallback.update(
                {
                    "contour": local_quad,
                    "area": area,
                    "epsilon": local_eps,
                    "source": "initial",
                    "params": None,
                }
            )

        # Enforce area requirement for primary detection
        if area < min_area:
            if debug_mode:
                print(
                    f"  Skipping small contour with area {area:.0f} (min required: {min_area:.0f})"
                )
            continue

        # If it's a quad and meets area, accept immediately
        if local_quad is not None:
            screen_contour = local_quad
            if debug_mode:
                print(
                    f"  Found 4-vertex contour with area {area:.0f} and epsilon factor {local_eps}"
                )
            break

    if screen_contour is None:
        print("No document contour found with initial parameters!")
        print("Trying alternative edge detection parameters...")

        # Try with different Canny parameters
        alternative_params = [
            (50, 150),  # Lower thresholds
            (100, 250),  # Higher thresholds
            (30, 100),  # Much lower thresholds
            (150, 300),  # Much higher thresholds
            (20, 80),  # Very low thresholds for weak edges
            (40, 120),  # Low-medium thresholds
            (60, 180),  # Medium thresholds
            (80, 240),  # Medium-high thresholds
        ]

        # Track alternative-pass best fallback as well (continue using best_fallback)
        # buffer for alt edge images to save later into debug folder
        debug_alt_images = []
        for i, (low, high) in enumerate(alternative_params):
            print(f"  Trying Canny({low}, {high})...")
            alt_edged = cv2.Canny(gray, low, high)

            if debug_mode:
                debug_alt_images.append((i + 1, alt_edged))

            try:
                alt_contours, _ = cv2.findContours(
                    alt_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(alt_contours) > 0:
                    alt_contours = sorted(
                        alt_contours, key=cv2.contourArea, reverse=True
                    )[:5]
                    for c in alt_contours:
                        contour_area = cv2.contourArea(c)

                        # Approximate to quadrilateral across candidates
                        peri = cv2.arcLength(c, True)
                        local_quad = None
                        local_eps = None
                        for epsilon_factor in epsilon_candidates:
                            approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)
                            if len(approx) == 4:
                                local_quad = approx
                                local_eps = epsilon_factor
                                break

                        # Update best fallback if this quad is the largest so far
                        if (
                            local_quad is not None
                            and contour_area > best_fallback["area"]
                        ):
                            best_fallback.update(
                                {
                                    "contour": local_quad,
                                    "area": contour_area,
                                    "epsilon": local_eps,
                                    "source": "alternative",
                                    "params": (low, high),
                                }
                            )

                        # Enforce area requirement for primary detection (configurable)
                        if contour_area < min_area:
                            print(
                                f"  Skipping small contour (area: {contour_area:.0f} < minimum: {min_area:.0f})"
                            )
                            continue

                        # Accept immediately if meets area and is a quad
                        if local_quad is not None:
                            screen_contour = local_quad
                            print(
                                f"  ✓ Found 4-sided contour with parameters ({low}, {high}) and epsilon {local_eps}, area: {contour_area:.0f}"
                            )
                            break
                    if screen_contour is not None:
                        break
            except cv2.error as e:
                print(f"    Error with parameters ({low}, {high}): {e}")
                continue

            if screen_contour is not None:
                break

    # Final fallback: if nothing met the min_area but we found a valid quadrilateral,
    # use the largest one above a smaller safety threshold, with a warning to the user.
    if screen_contour is None and best_fallback["contour"] is not None:
        if best_fallback["area"] >= fallback_min_area_arg:
            print("\n⚠ No contour met the 1000px minimum.")
            src = best_fallback["source"]
            eps = best_fallback["epsilon"]
            params = best_fallback["params"]
            if params is not None:
                print(
                    f"  Using largest quadrilateral found: {best_fallback['area']:.0f} px, from {src} pass, Canny{params}, epsilon {eps}"
                )
            else:
                print(
                    f"  Using largest quadrilateral found: {best_fallback['area']:.0f} px, from {src} pass, epsilon {eps}"
                )
            print(
                "  Note: This may be a partial document or a small region. Check the output."
            )
            screen_contour = best_fallback["contour"]

    if screen_contour is None:
        print("Error: Could not find a rectangular document contour!")
        print("Make sure the document has clear, straight edges and good contrast.")
        sys.exit(1)

    print("Document contour found! Processing...")

    # Apply the four point transform to obtain a top-down view of the original image
    contour_points = screen_contour.reshape(4, 2) * ratio

    # Guard against extremely small selected regions: if the selected quadrilateral
    # covers too little of the resized image, prefer a full-frame fallback to avoid
    # blank/meaningless results.
    # Default initialization for debug safety
    selected_area = 0.0
    area_fraction = 0.0
    try:
        small_region_fallback = False
        resized_total_area = float(image.shape[0] * image.shape[1])
        selected_area = float(cv2.contourArea(screen_contour))
        area_fraction = selected_area / (resized_total_area + 1e-6)
        # Threshold: if selected region < 4% of resized image, it's likely noise
        if area_fraction < min_area_frac_arg:
            print(
                f"⚠ Selected region is very small: {area_fraction*100:.2f}% of image. Using full-frame fallback."
            )
            small_region_fallback = True
    except Exception:
        small_region_fallback = False

    try:
        # Try to warp the original image
        if orig.shape[0] == 0 or orig.shape[1] == 0:
            print("Error: Original image has invalid dimensions!")
            sys.exit(1)

        if small_region_fallback:
            warped = orig.copy()
        else:
            warped = four_point_transform(orig, contour_points)

        if warped is None or warped.shape[0] == 0 or warped.shape[1] == 0:
            print("Error: Perspective transformation failed!")
            sys.exit(1)

        print("Perspective correction completed!")
        print(f"Warped image shape: {warped.shape}")
        print(f"Warped image range: {warped.min()}-{warped.max()}")

        # Process the warped image with high quality enhancements
        output_dir = create_output_directory(output_path)
        debug_dir = f"{output_dir}/debug_processing" if debug_mode else None
        quality_dir = f"{output_dir}/quality_comparison"

        # Create quality comparison directory
        os.makedirs(quality_dir, exist_ok=True)

        (
            enhanced_img,
            otsu_thresh,
            adaptive_mean_thresh,
            adaptive_gaussian_thresh,
            niblack_thresh,
            final_processed,
        ) = create_high_quality_scan(warped)

        # Save all quality versions with enhanced comparison
        save_all_quality_versions(
            output_dir,
            enhanced_img,
            otsu_thresh,
            adaptive_mean_thresh,
            adaptive_gaussian_thresh,
            niblack_thresh,
            final_processed,
            warped,  # perspective corrected
        )

        # Build named variants for selection
        variants = {
            "combined": final_processed,
            "adaptive-gaussian": adaptive_gaussian_thresh,
            "adaptive-mean": adaptive_mean_thresh,
            "otsu": otsu_thresh,
            "niblack": niblack_thresh,
            "grayscale": enhanced_img,
            "original": warped,
        }
        # Default to combined as a safe baseline
        best_img = final_processed
        chosen_variant_name = "combined"

        # Choose RECOMMENDED based on preference
        if prefer_variant != "auto":
            best_img = variants.get(prefer_variant)
            if (
                best_img is None
                or getattr(best_img, "size", 0) == 0
                or is_near_blank(best_img)
            ):
                # fall back to auto if chosen variant is invalid/blank
                prefer_variant = "auto"
            else:
                chosen_variant_name = prefer_variant

        if prefer_variant == "auto":
            best_img, best_score = None, -1.0
            chosen_variant_name = "combined"
            for name, img_candidate in variants.items():
                if img_candidate is None or getattr(img_candidate, "size", 0) == 0:
                    continue
                if is_near_blank(img_candidate):
                    continue
                score = content_score(img_candidate)
                # Apply profile-based biases
                if doc_type_profile == "table":
                    # Favor strong black/white and clear lines
                    if name in ("combined", "otsu", "adaptive-gaussian"):
                        score += 0.10
                    if name in ("grayscale", "original"):
                        score -= 0.05
                elif doc_type_profile == "text":
                    # Favor smoother grayscale to avoid jaggy edges
                    if name in ("grayscale",):
                        score += 0.12
                    if name in ("combined", "otsu"):
                        score -= 0.08
                if score > best_score:
                    best_img, best_score = img_candidate, score
                    chosen_variant_name = name
            if best_img is None:
                best_img = final_processed
                chosen_variant_name = "combined"

        if debug_mode and debug_dir:
            # Save a simple overlay showing detected contour on the resized image
            overlay = image.copy()
            try:
                cv2.drawContours(
                    overlay, [screen_contour.astype(int)], -1, (0, 255, 0), 2
                )
                cv2.imwrite(f"{debug_dir}/debug_detected_contour.jpg", overlay)
            except Exception:
                pass

        # Save debug images if debug mode is enabled
        if debug_mode and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/debug_01_resized.jpg", image)
            cv2.imwrite(f"{debug_dir}/debug_02_gray.jpg", gray)
            cv2.imwrite(f"{debug_dir}/debug_03_edges.jpg", edged)
            cv2.imwrite(f"{debug_dir}/debug_04_warped.jpg", warped)
            # Save alternative edge images captured earlier
            try:
                for idx, alt_img in debug_alt_images:
                    cv2.imwrite(f"{debug_dir}/debug_alt_edges_{idx}.jpg", alt_img)
            except Exception:
                pass
            # Save region stats
            try:
                with open(
                    f"{debug_dir}/debug_region_info.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(
                        f"selected_area_px_resized={selected_area:.1f}\narea_fraction={area_fraction:.6f}\nsmall_region_fallback={small_region_fallback}\n"
                    )
            except Exception:
                pass
            print(f"🔧 Debug processing images saved to: {debug_dir}/")

        # Also save main outputs in the root directory for convenience
        try:
            # Save the recommended version and grayscale in main folder
            if best_img is None or getattr(best_img, "size", 0) == 0:
                best_img = final_processed
            cv2.imwrite(f"{output_dir}/RECOMMENDED_scanned_document.jpg", best_img)
            cv2.imwrite(f"{output_dir}/GRAYSCALE_enhanced.jpg", enhanced_img)

            print(f"\n🎉 SUCCESS! All outputs saved to: {output_dir}/")
            print("=" * 60)
            print("📁 QUICK ACCESS FILES:")
            print("   • RECOMMENDED_scanned_document.jpg (main result)")
            print(f"     ↳ Variant used: {chosen_variant_name}")
            print("   • GRAYSCALE_enhanced.jpg (enhanced grayscale)")
            print("📂 QUALITY COMPARISON:")
            print(f"   • {quality_dir}/ (all 7 versions + selection guide)")
            if debug_mode and debug_dir:
                print("🔧 DEBUG INFO:")
                print(f"   • {debug_dir}/ (processing steps)")
            print(
                "\n💡 TIP: Check the quality_comparison folder to pick your favorite!"
            )

        except Exception as e:
            print(f"Error saving output files: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during perspective transformation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
