[![Coin Detection Example](https://github.com/azizmeca1/coin-detection-yolo/main/cpcoin_model/images/Capture d'écran 2025-12-27 130747.png?raw=true)
](https://github.com/azizmeca1/coin-detection-yolo/blob/main/cpoin_model/images/Capture%20d%27%C3%A9cran%202025-12-27%20130747.png?raw=true)

# Coin Detection and Counting with YOLO

A real-time coin detection and counting system using YOLOv8 for video analysis. This project implements object detection to identify and count coins in video streams with high accuracy.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Training](#model-training)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time coin detection** in video streams
- **Automatic counting** with maximum coin tracking
- **Visual bounding boxes** with confidence scores
- **Center point visualization** for each detected coin
- **Coordinate display** for precise localization
- **Information panel** showing current and maximum coin count
- **Adjustable display size** for different screen resolutions
- **High accuracy** detection with customizable confidence threshold

## 🎥 Demo

The system processes video frames and displays:
- Green bounding boxes around detected coins
- Confidence scores for each detection
- Red center points marking coin positions
- Real-time coin count and maximum count tracker

## 📦 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- Webcam or video file for testing

### Python Dependencies

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
```

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/azizmeca1/coin-detection-yolo.git
cd coin-detection-yolo
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download or place your trained model**

Ensure your trained model weights are in:
```
runs/detect/cpoin_model/weights/best.pt
```

## 🎓 Model Training

### Dataset Preparation

1. **Collect coin images**
   - Gather diverse images with different lighting conditions
   - Include various angles and backgrounds
   - Recommended: 500-1000+ images for good performance

2. **Annotate your dataset**
   - Use tools like [Roboflow](https://roboflow.com/), [LabelImg](https://github.com/heartexlabs/labelImg), or [CVAT](https://www.cvat.ai/)
   - Label each coin with bounding boxes
   - Export in YOLO format

3. **Organize your dataset structure**

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

4. **Create data.yaml configuration**

```yaml
train: ../dataset/train/images
val: ../dataset/valid/images

nc: 1  # number of classes
names: ['coin']  # class names
```

### Training Process

1. **Train the YOLOv8 model**

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano model for faster training
# or YOLO('yolov8s.pt')     # small model for better accuracy
# or YOLO('yolov8m.pt')     # medium model for even better accuracy

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='cpoin_model',
    patience=50,
    save=True,
    device=0  # Use GPU 0, or 'cpu' for CPU training
)
```

2. **Training parameters explanation**
   - `epochs`: Number of training iterations (100-300 recommended)
   - `imgsz`: Image size for training (640 is standard, 1280 for better accuracy)
   - `batch`: Batch size (adjust based on GPU memory)
   - `patience`: Early stopping patience
   - `device`: GPU device (0, 1, etc.) or 'cpu'

3. **Monitor training**
   - View training progress in terminal
   - Check TensorBoard logs: `tensorboard --logdir runs/detect/cpoin_model`
   - Training results saved in `runs/detect/cpoin_model/`

4. **Model evaluation**

```python
# Validate the model
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

### Model Export (Optional)

Export for different platforms:

```python
# Export to ONNX
model.export(format='onnx')

# Export to TensorRT (for NVIDIA GPUs)
model.export(format='engine')

# Export to CoreML (for Apple devices)
model.export(format='coreml')
```

## 💻 Usage

### Basic Usage

```bash
python detect_coins.py
```

### Using with Custom Video

Edit the video source in `detect_coins.py`:

```python
# For video file
cap = cv2.VideoCapture('path/to/your/video.mp4')

# For webcam
cap = cv2.VideoCapture(0)

# For IP camera
cap = cv2.VideoCapture('rtsp://your-camera-ip/stream')
```

### Adjusting Detection Parameters

Modify these parameters in the script:

```python
results = model.predict(
    frame,
    conf=0.25,      # Confidence threshold (0.1-0.9)
    iou=0.3,        # IoU threshold for NMS
    imgsz=1280,     # Image size for inference
    verbose=False
)
```

### Adjusting Display Size

Change the scale percentage:

```python
scale_percent = 50  # Reduce to 50% of original size
# 100 = original size, 50 = half size, 25 = quarter size
```

## 📁 Project Structure

```
coin-detection-yolo/
│
├── detect_coins.py           # Main detection script
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── dataset/                  # Training dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── runs/                     # Training and detection results
│   └── detect/
│       └── cpoin_model/
│           └── weights/
│               ├── best.pt   # Best model weights
│               └── last.pt   # Last checkpoint
│
├── videos/                   # Test videos
│   └── coin_vd_1.mp4
│
└── results/                  # Output results (optional)
    └── annotated_videos/
```

## 🔍 How It Works

### Detection Pipeline

1. **Video Capture**: Load video file or camera stream
2. **Frame Processing**: Extract individual frames
3. **Model Inference**: Run YOLO detection on each frame
4. **Post-processing**: 
   - Apply confidence threshold filtering
   - Non-maximum suppression (NMS) to remove duplicate detections
5. **Visualization**:
   - Draw bounding boxes
   - Add labels with confidence scores
   - Mark center points
   - Display coordinates
6. **Counting**: Track and display current and maximum coin count
7. **Display**: Show processed frame with annotations

### Detection Parameters

- **Confidence Threshold (conf)**: Minimum confidence score for detection (0.25 = 25%)
- **IoU Threshold (iou)**: Overlap threshold for NMS (0.3 = 30%)
- **Image Size (imgsz)**: Input size for model inference (1280 for high accuracy)

## 📊 Results

### Expected Performance

- **Inference Speed**: 30-60 FPS on GPU (depends on hardware)
- **Accuracy**: mAP50 > 0.90 (with proper training)
- **Detection Range**: Works with various coin sizes and positions

### Sample Output

The system displays:
- Real-time bounding boxes around coins
- Confidence scores (e.g., "Piece 0.95")
- Center coordinates (e.g., "x:320 y:240")
- Current count: "Pieces: 5"
- Maximum count: "Max: 8"

## ⚙️ Configuration

### Performance Optimization

**For faster processing:**
```python
model.predict(frame, conf=0.3, iou=0.4, imgsz=640, half=True)
```

**For better accuracy:**
```python
model.predict(frame, conf=0.15, iou=0.3, imgsz=1280, augment=True)
```

### Color Customization

Change bounding box and text colors:

```python
# BGR color format
box_color = (0, 255, 0)      # Green
label_color = (0, 255, 0)    # Green
center_color = (0, 0, 255)   # Red
coord_color = (255, 255, 0)  # Cyan
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Low FPS or slow processing
- **Solution**: Reduce `imgsz` to 640 or enable GPU acceleration

**Issue**: Too many false detections
- **Solution**: Increase `conf` threshold (try 0.4-0.5)

**Issue**: Missing detections
- **Solution**: Decrease `conf` threshold or retrain with more data

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size during training or use smaller model

**Issue**: Model file not found
- **Solution**: Check path to `best.pt` file is correct

### Debug Mode

Enable verbose output for debugging:

```python
results = model.predict(frame, verbose=True)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- OpenCV for computer vision utilities
- The open-source community for various tools and libraries

## 📧 Contact

Your Name - [@azizmeca1](https://github.com/azizmeca1)

Project Link: [https://github.com/azizmeca1/coin-detection-yolo](https://github.com/azizmeca1/coin-detection-yolo)

---

**Note**: Make sure to update the model path, video path, and parameters according to your specific setup and requirements.# coin-detection-yolo
YOLO-based coin detection and counting system for video analysis
