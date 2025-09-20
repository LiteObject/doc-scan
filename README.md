# Document Scanner

A Python-based document scanner that automatically detects document boundaries, applies perspective correction, and generates high-quality scanned images using OpenCV.

## Features

- **Automatic Document Detection**: Uses advanced contour detection to identify document boundaries
- **Perspective Correction**: Applies four-point transformation for proper document alignment
- **Multiple Quality Options**: Generates 7 different processing versions for optimal results
- **Enhanced Image Processing**: Includes noise reduction, contrast enhancement, and sharpening
- **Flexible Output**: Supports custom output directories with organized file structure
- **Debug Mode**: Optional intermediate image saving for troubleshooting
- **Command Line Interface**: Easy-to-use CLI with comprehensive options
- **Robust Fallbacks**: Avoids blank outputs by using a full-frame fallback when the detected region is too small
- **RECOMMENDED Control**: Choose which processed variant is saved as RECOMMENDED via `--prefer`
 - **Profiles**: Bias auto selection for tables vs text via `--doc-type`

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV (cv2)
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LiteObject/doc-scan.git
cd doc-scan
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

```bash
python scanner.py input_image.jpg
```

### Advanced Options

```bash
# Specify custom output directory
python scanner.py document.jpg --output ./my_scans

# Enable debug mode
python scanner.py document.jpg --debug

# Combine options
python scanner.py document.jpg --output ./scans --debug

# Tune detection thresholds
python scanner.py document.jpg --min-area 1500 --fallback-min-area 600 --min-area-frac 0.05

# Choose the RECOMMENDED variant
# Force clean binary (default behavior)
python scanner.py document.jpg --prefer combined

# Use enhanced grayscale as recommended
python scanner.py document.jpg --prefer grayscale

# Let the tool auto-select based on content scoring
python scanner.py document.jpg --prefer auto

# Bias auto selection for tables (crisp B/W lines) or text (smoother)
python scanner.py document.jpg --prefer auto --doc-type table
python scanner.py document.jpg --prefer auto --doc-type text
```

### Command Line Arguments

- `input_file`: Path to the input image file (required)
- `--output, -o`: Custom output directory path (optional)
- `--debug`: Enable debug mode to save intermediate processing images (optional)
- `--min-area`: Minimum contour area (in resized-pixels) to accept as the document (default: 1000)
- `--fallback-min-area`: Minimum area to allow a fallback quadrilateral if no primary match is found (default: 500)
- `--min-area-frac`: If the selected quadrilateral covers less than this fraction of the resized image, use full-frame fallback (default: 0.04 = 4%)
- `--prefer`: Which variant to save as RECOMMENDED. Options: `combined` (default), `grayscale`, `original`, `otsu`, `adaptive-mean`, `adaptive-gaussian`, `niblack`, `auto` (score-based)
- `--doc-type`: Biases auto selection. Options: `auto` (default), `table` (favor crisp B/W and structured edges), `text` (favor smoother grayscale)
- `--help, -h`: Show help message and usage examples

## Output Structure

The scanner creates an organized directory structure with multiple quality options:

```
output_directory/
├── scanned_output_YYYYMMDD_HHMMSS/
│   ├── RECOMMENDED_scanned_document.jpg  # Main result
│   ├── GRAYSCALE_enhanced.jpg           # Enhanced grayscale version
│   ├── quality_comparison/              # All processing versions
│   │   ├── 00_original_perspective_corrected.jpg
│   │   ├── 01_enhanced_grayscale.jpg
│   │   ├── 02_otsu_threshold.jpg
│   │   ├── 03_adaptive_mean.jpg
│   │   ├── 04_adaptive_gaussian.jpg
│   │   ├── 05_niblack_local.jpg
│   │   ├── 06_combined_optimized.jpg
│   │   └── README.txt               # Selection guide
│   └── debug_processing/            # Debug images (if --debug enabled)
│       ├── debug_01_resized.jpg
│       ├── debug_02_gray.jpg
│       ├── debug_03_edges.jpg
│       ├── debug_04_warped.jpg
│       ├── debug_detected_contour.jpg
│       ├── debug_alt_edges_*.jpg
│       └── debug_region_info.txt    # Area stats and fallback info
```

## Quality Processing Options

The scanner generates multiple versions using different image processing techniques:

1. **Original Perspective Corrected**: Document after perspective transformation only
2. **Enhanced Grayscale**: Noise reduction, contrast enhancement, and sharpening applied
3. **Otsu Threshold**: Black and white using automatic threshold detection
4. **Adaptive Mean**: Black and white using adaptive mean thresholding
5. **Adaptive Gaussian**: Black and white using adaptive Gaussian thresholding
6. **Niblack Local**: Black and white using Niblack-like local thresholding
7. **Combined Optimized**: Recommended version with morphological cleanup

## Image Processing Pipeline

1. **Preprocessing**: Resize, convert to grayscale, apply Gaussian blur
2. **Edge Detection**: Canny edge detection with multiple parameter sets
3. **Contour Detection**: Find and analyze document boundaries
4. **Perspective Correction**: Four-point transformation to correct document perspective
5. **Quality Enhancement**: Apply denoising, CLAHE, and unsharp masking
6. **Thresholding**: Multiple techniques for optimal text/background separation
7. **Post-processing**: Morphological operations for cleanup

## Error Handling

The scanner includes comprehensive error handling for:

- File not found errors
- Invalid image formats
- Image loading failures
- Contour detection issues
- Perspective transformation problems
- File I/O errors

## Debug Mode

Enable debug mode with `--debug` to save intermediate processing images:

- Resized input image, grayscale, and initial edges
- Alternative edge images across multiple Canny thresholds
- Detected contour overlay
- Warped image (or full-frame fallback) and region stats

### Full-frame fallback (anti-blank safeguard)

When the detected quadrilateral covers less than a configurable fraction of the resized image (`--min-area-frac`, default 4%), the scanner skips perspective warp and processes the full original frame. This prevents blank or near-blank outputs from tiny/noisy contours.

## Technical Details

### Dependencies

- **OpenCV (cv2)**: Computer vision and image processing
- **NumPy**: Array operations and mathematical computations
- **argparse**: Command line argument parsing
- **datetime**: Timestamp generation for output folders
- **os/sys**: File system operations and system interactions

### Key Algorithms

- **Four-Point Transformation**: Perspective correction using homography
- **Adaptive Thresholding**: Multiple techniques for varying lighting conditions
- **Non-Local Means Denoising**: Advanced noise reduction
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Morphological Operations**: Image cleanup and enhancement

## Troubleshooting

### Common Issues

1. **"No contours found"**: Ensure the document has clear edges and good contrast
2. **"No rectangular contour found"**: Try with better lighting or clearer document boundaries
3. **Poor scan quality or harsh-looking “recommended”**:
	- The scanner scores all variants and skips near-blank candidates.
	- If the recommended looks too bold/harsh, try `--prefer grayscale` or keep `--prefer combined` and adjust brightness/contrast in a viewer.
	- If results still look weak, raise `--min-area-frac` (e.g., 0.06–0.1) to force full-frame processing more often, or increase `--min-area` (e.g., 1500–3000).

Note: The console prints which variant was saved as RECOMMENDED (for example, `Variant used: combined`).
4. **File permission errors**: Ensure write permissions for the output directory

### Tips for Better Results

- Use good lighting with minimal shadows
- Ensure the document has clear, straight edges
- Place the document on a contrasting background
- Keep the camera/phone steady when taking the photo
- Avoid reflections and glare on the document surface

## License

This project is open source. Please check the license file for specific terms.

## Future Enhancements

Potential improvements for future versions:

- Batch processing for multiple documents
- GUI interface for easier use
- Additional image enhancement algorithms
- Support for different output formats (PDF, TIFF)
- Configuration file support
- Performance optimizations for large images