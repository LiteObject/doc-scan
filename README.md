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
```

### Command Line Arguments

- `input_file`: Path to the input image file (required)
- `--output, -o`: Custom output directory path (optional)
- `--debug`: Enable debug mode to save intermediate processing images (optional)
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
│       └── debug_03_edges.jpg
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

- Resized input image
- Grayscale conversion
- Edge detection results
- Alternative edge detection attempts (if needed)

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
3. **Poor scan quality**: Check the quality_comparison folder for alternative versions
4. **File permission errors**: Ensure write permissions for the output directory

### Tips for Better Results

- Use good lighting with minimal shadows
- Ensure the document has clear, straight edges
- Place the document on a contrasting background
- Keep the camera/phone steady when taking the photo
- Avoid reflections and glare on the document surface

## Code Quality

The project follows Python best practices:

- Type hints for better code documentation
- Comprehensive error handling
- Modular function design
- Clear variable naming conventions
- Detailed docstrings and comments
- Lint-compliant code (pylint)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure code follows the existing style
6. Submit a pull request

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