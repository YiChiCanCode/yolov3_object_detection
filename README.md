## YOLOv3 Object Detection**

This repository contains code for YOLOv3-based object detection, which is an advanced deep learning model capable of detecting and localizing multiple objects in images.


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Weights](#model-and-weights)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

YOLO (You Only Look Once) is a real-time object detection algorithm that processes the entire image in one forward pass through the neural network. YOLOv3 is an improvement over its predecessors, providing better detection accuracy and multi-scale detection capabilities.

This repository implements YOLOv3 for object detection tasks using Python and the PyTorch deep learning framework.

## Requirements

- Python 3.6 or later
- PyTorch 1.7 or later
- NumPy
- OpenCV (optional, for image processing and visualization)
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YiChiCanCode/yolov3_object_detection.git
cd yolov3_object_detection
```

2. Install the required dependencies:

## Usage
The program will capture images using cv2 and apply object detection algorithm on the frames.

```bash
python yolov.py --image path/to/your/image.jpg
```

You could also modify your code based on your need to detect objects for images or videos. 

## Model and Weights

Pre-trained YOLOv3 weights and configuration files are available in the directory. You can also download the original YOLOv3 weights from the official YOLO website.

## Results

Example Results are as below:


## Contributing

Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

