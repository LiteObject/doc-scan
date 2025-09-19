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
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    enhanced_img = enhance_image_quality(gray)

    # Get all threshold versions
    otsu_thresh, adaptive_mean_thresh, adaptive_gaussian_thresh, niblack_thresh = (
        advanced_threshold(enhanced_img)
    )

    # Create final optimized version by combining techniques
    # Use adaptive Gaussian as base and refine with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_processed = cv2.morphologyEx(
        adaptive_gaussian_thresh, cv2.MORPH_CLOSE, kernel
    )
    final_processed = cv2.morphologyEx(final_processed, cv2.MORPH_OPEN, kernel)

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
        cv2.imwrite(filepath, img_data)
        print(f"   ✓ {filename}")

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
    return parser.parse_args()


def main():
    """Main function to handle CLI arguments and run the scanner."""
    args = parse_arguments()

    input_file = args.input_file
    debug_mode = args.debug
    output_path = args.output

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
    edged = cv2.Canny(gray, 75, 200)

    # Save debug images if debug mode is enabled
    if debug_mode:
        cv2.imwrite("debug_01_resized.jpg", image)
        cv2.imwrite("debug_02_gray.jpg", gray)
        cv2.imwrite("debug_03_edges.jpg", edged)
        print("Debug: Saved intermediate processing images")

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

    screen_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_contour = approx
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
        ]

        for i, (low, high) in enumerate(alternative_params):
            print(f"  Trying Canny({low}, {high})...")
            alt_edged = cv2.Canny(gray, low, high)

            if debug_mode:
                cv2.imwrite(f"debug_alt_edges_{i+1}.jpg", alt_edged)

            try:
                alt_contours, _ = cv2.findContours(
                    alt_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(alt_contours) > 0:
                    alt_contours = sorted(
                        alt_contours, key=cv2.contourArea, reverse=True
                    )[:5]
                    for c in alt_contours:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) == 4:
                            screen_contour = approx
                            print(
                                f"  ✓ Found 4-sided contour with parameters ({low}, {high})"
                            )
                            break
            except cv2.error as e:
                print(f"    Error with parameters ({low}, {high}): {e}")
                continue

            if screen_contour is not None:
                break

    if screen_contour is None:
        print("Error: Could not find a rectangular document contour!")
        print("Make sure the document has clear, straight edges and good contrast.")
        sys.exit(1)

    print("Document contour found! Processing...")

    # Apply the four point transform to obtain a top-down view of the original image
    contour_points = screen_contour.reshape(4, 2) * ratio

    try:
        # Try to warp the original image
        if orig.shape[0] == 0 or orig.shape[1] == 0:
            print("Error: Original image has invalid dimensions!")
            sys.exit(1)

        warped = four_point_transform(orig, contour_points)

        if warped is None or warped.shape[0] == 0 or warped.shape[1] == 0:
            print("Error: Perspective transformation failed!")
            sys.exit(1)

        print("Perspective correction completed!")

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

        # Save debug images if debug mode is enabled
        if debug_mode and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/debug_01_resized.jpg", image)
            cv2.imwrite(f"{debug_dir}/debug_02_gray.jpg", gray)
            cv2.imwrite(f"{debug_dir}/debug_03_edges.jpg", edged)
            print(f"🔧 Debug processing images saved to: {debug_dir}/")

        # Also save main outputs in the root directory for convenience
        try:
            # Save the recommended version and grayscale in main folder
            cv2.imwrite(
                f"{output_dir}/RECOMMENDED_scanned_document.jpg", final_processed
            )
            cv2.imwrite(f"{output_dir}/GRAYSCALE_enhanced.jpg", enhanced_img)

            print(f"\n🎉 SUCCESS! All outputs saved to: {output_dir}/")
            print("=" * 60)
            print("📁 QUICK ACCESS FILES:")
            print("   • RECOMMENDED_scanned_document.jpg (main result)")
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
