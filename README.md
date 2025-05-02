# Football Object Detection

A computer vision project for detecting and tracking players, the ball, and calculating various metrics in football videos.

## Features

- Player detection and tracking using YOLOv8
- Team classification based on jersey colors
- Ball tracking with Kalman filtering
- Field line detection
- Player speed calculation
- Video output with annotations

## Project Structure

- `main.py`: Core processing script with all object detection and tracking
- `modules/`: Helper modules
  - `field_detector.py`: Detects football field lines and boundaries
  - `model_provider.py`: Singleton provider for YOLO model loading
  - `speed_calculator.py`: Calculates player speeds and other metrics
- `weights/`: Pretrained YOLOv8 model weights
- `test_videos/`: Sample videos for testing
- `output/`: Directory for processed video output
- `ModelTraining.ipynb`: Jupyter notebook with model training process

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv8
- filterpy
- scikit-learn
- pandas

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Football-Object-Detection.git
   cd Football-Object-Detection
   ```

2. Install dependencies:
   ```
   pip install ultralytics opencv-python numpy filterpy scikit-learn pandas
   ```

3. Download the model weights (or train your own using ModelTraining.ipynb)

## Usage

Process a video file:

```python
python main.py --video path/to/your/video.mp4
```

## Model Training

The model was trained using YOLOv8 on a dataset of football images. The training process and hyperparameters are documented in `ModelTraining.ipynb`.

## Examples

Output videos will be saved in the `output/` directory with annotations showing:
- Player bounding boxes with team classification
- Ball tracking
- Player speeds
- Field boundaries

