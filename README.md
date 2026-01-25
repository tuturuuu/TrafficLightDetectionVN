# Traffic Light Detection with YOLO

## Project Overview

This project implements an enhanced YOLO-based traffic light detection system optimized for Vietnamese traffic environments. The system is designed to accurately detect traffic lights and their countdown timers in real-world driving scenarios, enabling autonomous vehicles and intelligent traffic monitoring systems to interpret traffic signals reliably.

### Key Features

- YOLO base model with custom enhancements for improved detection accuracy
- CBAM (Convolutional Block Attention Module) integration for focused feature learning
- Tiling-based inference for handling images at various scales and resolutions
- Comprehensive utilities for data conversion and preprocessing

## Requirements

- Python 3.12.2
- CUDA Toolkit (recommended for GPU acceleration)
- pip (Python package manager)

## Environment Setup

### Step 1: Verify Python Version

Ensure you have Python 3.12.2 installed:

```bash
python --version
```

Expected output: `Python 3.12.2`

If you need to install Python 3.12.2, download it from [python.org](https://www.python.org/downloads/) or use your system's package manager.

### Step 2: Create Virtual Environment

Create an isolated Python virtual environment:

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

On Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### Step 4: Upgrade pip

Ensure pip is up to date:

```bash
pip install --upgrade pip
```

### Step 5: Install Dependencies

Install all required packages:

```bash
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-deps
```

This will install:

- **ultralytics** (8.3.230) - YOLO framework
- **torch** (2.9.1+cu128) - Deep learning framework with CUDA support
- **opencv-python** (4.12.0) - Computer vision operations
- **matplotlib** (3.10.7) - Visualization
- **tqdm** (4.67.1) - Progress tracking
- **PyYAML** (6.0.3) - Configuration file handling

### Step 6: Verify Installation

Verify that all packages are installed correctly:

```bash
python -c "import torch; import cv2; import ultralytics; print('All packages installed successfully')"
```

## Project Structure

```
.
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── custom_modules.py                  # Custom module implementations
├── yolo_tiling.py                     # YOLO model with tiling inference
├── yolo_tiling_cbam.py                # YOLO with CBAM attention module
│
├── weights/                           # The weights of the model
│   ├── best.pt                        # Sample model weight
│
├── evaluation/                        # Evaluation and analysis utilities
│   ├── evaluation.py                  # Main evaluation files
│   └── false_examples.py              # False positive analysis
│
└── utils/                             # Utility functions and helpers
    ├── convert_bstld_to_yolo.py       # Dataset conversion utility (BSTLD to YOLO format)
    ├── tiling.py                      # Image tiling utilities for multi-scale inference
    └── print_layers/                  # Layer inspection and debugging tools
```

### Directory Details

#### Root Level

- **custom_modules.py**: Contains custom PyTorch modules and layer implementations used across the project
- **yolo_tiling.py**: Main detection model using YOLOv with tiling-based inference for improved performance on images of varying sizes
- **yolo_tiling_cbam.py**: Enhanced model variant incorporating CBAM (Convolutional Block Attention Module) for attention-based feature refinement

#### evaluation/

- **evaluation.py**: Comprehensive evaluation framework for measuring model performance, including metrics calculation and result visualization
- **false_examples.py**: Tools for identifying, analyzing, and visualizing false positive detections to identify model weaknesses

#### utils/

- **convert_bstld_to_yolo.py**: Conversion utility for transforming BSTLD (Berlin Traffic Light Dataset) format annotations to YOLO format for model training
- **tiling.py**: Image processing utilities for implementing tiling-based inference, enabling detection across multi-scale features
- **print_layers/**: Debugging utilities for inspecting model architecture and layer information

## Usage

### Basic Detection

Load the model and run predictions:

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
results = model.predict(source='path/to/image.jpg')
```

The model automatically handles image tiling internally. For detection, the image is divided into tiles, YOLOv11 inference is run on each tile, and detection results are combined and merged to produce the final output. This approach ensures traffic lights are detected at multiple scales and resolutions.

### Model Evaluation

Configure the evaluation parameters in [evaluation/evaluation.py](evaluation/evaluation.py) and run it directly:

```bash
python evaluation/evaluation.py
```

## Notes

- For optimal performance, GPU with CUDA support is recommended
- The CBAM-enhanced model may provide better accuracy for challenging lighting conditions
- Tiling inference helps detect traffic lights at various scales within the same image

## Support

For issues or questions, please refer to the [ultralytics YOLOv11 documentation](https://docs.ultralytics.com/)
