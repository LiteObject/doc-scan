#!/usr/bin/env python3
"""
make_printable.py - Prepare documents for printing by removing background colors
while preserving text and images as much as possible.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def create_output_directory(base_dir="./output"):
    """Create timestamped output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"printable_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def remove_background_advanced(image, method="auto", threshold_value=None):
    """Advanced background removal supporting multiple strategies.

    Returns a tuple: (processed_image, mask)
    """
    # Convert to grayscale and denoise slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "auto":
        # Otsu auto threshold, adjusted by brightness
        threshold, _ = cv2.threshold(
            gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mean_brightness = float(np.mean(gray))
        if mean_brightness > 200:
            threshold = min(threshold + 20, 250)
        elif mean_brightness < 100:
            threshold = max(threshold - 20, 50)
    elif method == "light":
        threshold, _ = cv2.threshold(
            gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = min(threshold + 30, 240)
    elif method == "aggressive":
        threshold, _ = cv2.threshold(
            gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = max(threshold - 30, 100)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(
            gray_smooth,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        result = np.full_like(image, 255)
        result[mask > 0] = image[mask > 0]
        return result, mask
    elif method == "custom" and threshold_value is not None:
        threshold = int(threshold_value)
    else:
        threshold = 200

    # Foreground mask (invert background)
    _, mask = cv2.threshold(gray_smooth, threshold, 255, cv2.THRESH_BINARY_INV)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove tiny specks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result, mask


def remove_background_color_based(image):
    """Remove background using HSV color segmentation for colored pages."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    peak_flat_index = int(np.argmax(hist))
    peak_indices = np.unravel_index(peak_flat_index, hist.shape)
    peak_hue, peak_sat = int(peak_indices[0]), int(peak_indices[1])

    if peak_sat < 30:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        lower = np.array([max(0, peak_hue - 20), max(0, peak_sat - 50), 0])
        upper = np.array([min(179, peak_hue + 20), min(255, peak_sat + 50), 255])
        bg_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(bg_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]
    return result, mask


def remove_background(image, method="auto", threshold_value=None):
    """
    Remove background colors while preserving text and images.

    Args:
        image: Input image (BGR)
        method: Background removal method ('auto', 'light', 'aggressive', 'custom')
        threshold_value: Custom threshold value (0-255) when method='custom'

    Returns:
        Processed image with background removed, and mask used
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram to determine background color
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Find the dominant color (likely background)
    background_intensity = int(np.argmax(hist))

    if method == "auto":
        # Automatically determine threshold based on background
        if background_intensity > 200:  # Light background
            threshold = max(0, background_intensity - 30)
        elif background_intensity > 127:  # Medium background
            threshold = max(0, background_intensity - 20)
        else:  # Dark background
            threshold = min(255, background_intensity + 30)
    elif method == "light":
        threshold = max(180, background_intensity - 20)
    elif method == "aggressive":
        threshold = max(150, background_intensity - 40)
    elif method == "custom" and threshold_value is not None:
        threshold = int(threshold_value)
    else:
        threshold = 200  # Default fallback

    # Create mask for non-background areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Create white background
    result = np.full_like(image, 255)

    # Copy non-background pixels
    result[mask > 0] = image[mask > 0]

    return result, mask


def enhance_for_printing(image):
    """
    Enhance image specifically for printing.

    Args:
        image: Input image (BGR)

    Returns:
        Enhanced image optimized for printing
    """
    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)

    # Convert to LAB color space for better color/lightness control
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply stronger CLAHE to L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Slightly desaturate color channels for printing
    a_channel = (a_channel * 0.9).astype(np.uint8)
    b_channel = (b_channel * 0.9).astype(np.uint8)

    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Subtle sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Blend with original for subtle enhancement
    result = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)

    return result


def enhance_text_clarity(image):
    """Enhance text clarity with unsharp masking + CLAHE."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(unsharp)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def create_printable_versions(
    image_path, output_dir, method="auto", threshold=None, enhance=True, debug=False
):
    """
    Create multiple printable versions of the document.

    Args:
        image_path: Path to input image
        output_dir: Output directory path
        method: Background removal method
        threshold: Custom threshold value
        enhance: Whether to apply enhancement
        debug: Save debug images

    Returns:
        Paths to generated files
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Get original filename without extension
    filename = Path(image_path).stem

    generated_files = []

    # Create versions subdirectory
    versions_dir = os.path.join(output_dir, "versions")
    os.makedirs(versions_dir, exist_ok=True)

    # 1. Background removed version (advanced)
    print("Removing background (advanced method)...")
    if method == "color":
        bg_removed, mask = remove_background_color_based(image)
    else:
        bg_removed, mask = remove_background_advanced(image, method, threshold)
    bg_removed_path = os.path.join(
        versions_dir, f"01_{filename}_background_removed.jpg"
    )
    cv2.imwrite(bg_removed_path, bg_removed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    generated_files.append(bg_removed_path)
    print(f"  Saved: {os.path.basename(bg_removed_path)}")

    # 2. Enhanced version (if requested)
    if enhance:
        print("Creating enhanced version...")
        enhanced_img = enhance_for_printing(bg_removed)
        enhanced_path = os.path.join(versions_dir, f"02_{filename}_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated_files.append(enhanced_path)
        print(f"  Saved: {os.path.basename(enhanced_path)}")
    else:
        enhanced_img = bg_removed

    # 3. Text-optimized version
    print("Creating text-optimized version...")
    text_optimized = enhance_text_clarity(bg_removed)
    text_path = os.path.join(versions_dir, f"03_{filename}_text_optimized.jpg")
    cv2.imwrite(text_path, text_optimized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    generated_files.append(text_path)
    print(f"  Saved: {os.path.basename(text_path)}")

    # 4. High contrast black and white (adaptive)
    print("Creating high contrast B&W version...")
    gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
    bw_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    bw_path = os.path.join(versions_dir, f"04_{filename}_black_white.jpg")
    cv2.imwrite(bw_path, bw_adaptive, [cv2.IMWRITE_JPEG_QUALITY, 95])
    generated_files.append(bw_path)
    print(f"  Saved: {os.path.basename(bw_path)}")

    # 5. Grayscale version (from enhanced image for best quality)
    print("Creating grayscale version...")
    grayscale = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    grayscale_path = os.path.join(versions_dir, f"05_{filename}_grayscale.jpg")
    cv2.imwrite(grayscale_path, grayscale, [cv2.IMWRITE_JPEG_QUALITY, 95])
    generated_files.append(grayscale_path)
    print(f"  Saved: {os.path.basename(grayscale_path)}")

    # Save the recommended version (enhanced or background removed)
    recommended = enhanced_img
    recommended_path = os.path.join(output_dir, f"PRINTABLE_{filename}.jpg")
    cv2.imwrite(recommended_path, recommended, [cv2.IMWRITE_JPEG_QUALITY, 95])
    generated_files.append(recommended_path)
    print(f"\nRecommended version saved: {os.path.basename(recommended_path)}")

    # Debug images
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Save mask
        mask_path = os.path.join(debug_dir, "mask.jpg")
        cv2.imwrite(mask_path, mask)

        # Save original for comparison
        original_path = os.path.join(debug_dir, "original.jpg")
        cv2.imwrite(original_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Create and save comparison image (resize to manageable width)
        def resize_to_width(img, width=800):
            h, w = img.shape[:2]
            scale = width / float(w)
            return cv2.resize(img, (width, int(h * scale)))

        before = resize_to_width(image)
        after = resize_to_width(recommended)
        # Pad to same height
        max_h = max(before.shape[0], after.shape[0])

        def pad_to_height(img, target_h):
            pad = max(0, target_h - img.shape[0])
            if pad == 0:
                return img
            return cv2.copyMakeBorder(
                img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

        before = pad_to_height(before, max_h)
        after = pad_to_height(after, max_h)

        # Add labels above images
        label_h = 40
        before_labeled = cv2.copyMakeBorder(
            before, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            before_labeled,
            "Original",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        after_labeled = cv2.copyMakeBorder(
            after, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            after_labeled,
            "Printable",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        comparison = np.hstack([before_labeled, after_labeled])
        comparison_path = os.path.join(debug_dir, "before_after_comparison.jpg")
        cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"\nDebug images saved in: {debug_dir}")

    # Create README
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("PRINTABLE DOCUMENT VERSIONS\n")
        f.write("=" * 50 + "\n\n")
        f.write("This folder contains printer-friendly versions of your document.\n\n")
        f.write("FILES:\n")
        f.write("- PRINTABLE_*.jpg: Recommended version for printing\n")
        f.write("\nVERSIONS FOLDER:\n")
        f.write(
            "- 01_*_background_removed.jpg: Background removed with preserved colors\n"
        )
        f.write("- 02_*_enhanced.jpg: Enhanced contrast and sharpness (if enabled)\n")
        f.write("- 03_*_text_optimized.jpg: Optimized for text clarity\n")
        f.write("- 04_*_black_white.jpg: Pure black and white (maximum ink savings)\n")
        f.write("- 05_*_grayscale.jpg: Grayscale version (from enhanced)\n")
        f.write("\nTIPS FOR PRINTING:\n")
        f.write("- Use the PRINTABLE_* file for best results\n")
        f.write("- For text documents, try the black_white version\n")
        f.write("- For documents with images, use background_removed or enhanced\n")
        f.write("- Print in 'Draft' or 'Economy' mode to save even more ink\n")

    return generated_files


def estimate_ink_savings(original_path, processed_path):
    """
    Estimate approximate ink savings.

    Args:
        original_path: Path to original image
        processed_path: Path to processed image

    Returns:
        Estimated percentage of ink saved
    """
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

    if original is None or processed is None:
        return 0.0

    # Resize processed to original if sizes differ
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    # Calculate average darkness (0 = black, 255 = white)
    # Convert to numpy arrays explicitly for type checkers
    orig_avg = float(np.mean(np.asarray(original, dtype=np.float64)))
    proc_avg = float(np.mean(np.asarray(processed, dtype=np.float64)))

    # Estimate ink usage (darker = more ink)
    orig_ink = (255.0 - orig_avg) / 255.0
    proc_ink = (255.0 - proc_avg) / 255.0

    # Calculate savings
    if orig_ink > 0:
        savings = ((orig_ink - proc_ink) / orig_ink) * 100.0
        return max(0.0, savings)

    return 0.0


def main():
    """CLI entry point for creating printer-friendly document images."""
    parser = argparse.ArgumentParser(
        description="Make documents printer-friendly by removing background colors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s document.jpg
    %(prog)s document.jpg --method aggressive
    %(prog)s document.jpg --method adaptive
    %(prog)s document.jpg --method color
    %(prog)s document.jpg --threshold 210
    %(prog)s document.jpg --no-enhance --debug
    %(prog)s document.jpg --output ./printable_docs
        """,
    )

    parser.add_argument("input_file", help="Path to the input image file")

    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "light", "aggressive", "adaptive", "color", "custom"],
        default="auto",
        help="Background removal method (default: auto)",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        help="Custom threshold value (0-255) for method=custom",
    )

    parser.add_argument(
        "--no-enhance", action="store_true", help="Skip enhancement step"
    )

    parser.add_argument("--debug", action="store_true", help="Save debug images")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Validate custom threshold
    if args.method == "custom" and args.threshold is None:
        print("Error: --threshold required when using method=custom")
        sys.exit(1)

    if args.threshold is not None and (args.threshold < 0 or args.threshold > 255):
        print("Error: Threshold must be between 0 and 255")
        sys.exit(1)

    try:
        # Create output directory
        output_dir = create_output_directory(args.output)
        print(f"\nProcessing: {args.input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Method: {args.method}")
        if args.threshold is not None:
            print(f"Threshold: {args.threshold}")
        print()

        # Process the image
        generated_files = create_printable_versions(
            args.input_file,
            output_dir,
            method=args.method,
            threshold=args.threshold,
            enhance=not args.no_enhance,
            debug=args.debug,
        )

        # Estimate ink savings using the recommended output (last path)
        if generated_files:
            savings = estimate_ink_savings(args.input_file, generated_files[-1])
            if savings > 0:
                print(f"\nEstimated ink savings: ~{savings:.1f}%")

        print(
            f"\n\u2713 Successfully created {len(generated_files)} printable versions"
        )
        print(f"\u2713 Output saved to: {output_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
