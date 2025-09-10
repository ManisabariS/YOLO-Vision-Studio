
# VisionDetect Pro

[![Version](https://img.shields.io/badge/Version-2.0.0-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%252B-green.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%252B-orange.svg)]()
[![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)]()

![VisionDetect Pro Interface](https://via.placeholder.com/800x500/2c3e50/ffffff?text=VisionDetect+Pro+Interface)

**VisionDetect Pro** is an advanced real-time object detection application built with YOLOv8, designed for high-performance inspection and analysis tasks. It offers a user-friendly interface with extensive customization for camera settings, detection parameters, and output configurations.

---

## 🚀 Features

### 🎯 Advanced Object Detection
- Real-time object detection using YOLOv8 models.
- Customizable confidence and IoU thresholds.
- Class filtering for focused detection.
- Comprehensive detection statistics and logging.

### 📷 Camera Control
- Multi-camera support with automatic detection.
- Adjustable camera properties (exposure, gain, white balance, etc.).
- Resolution and FPS customization.
- Support for multiple camera backends (DirectShow, Media Foundation, V4L2).

### ⚡ Performance Optimization
- Multi-threaded processing for smooth operation.
- Frame skipping options for higher FPS.
- GPU acceleration support (CUDA).
- Half-precision (FP16) inference for faster processing.

### 📊 Analytics & Reporting
- Real-time FPS and inference time monitoring.
- Detection statistics and class distribution.
- CSV export of detection logs.
- Annotated frame capture with customizable quality.

### 🎨 User Experience
- Modern, dark-themed interface.
- Tab-based organization of settings.
- Real-time video feed with detection overlay.
- Settings import/export functionality.
- Comprehensive status information.

---

## 📦 Installation

### ✅ Prerequisites
- Python 3.8 or higher.
- Webcam or camera device.
- NVIDIA GPU (optional, for CUDA acceleration).

### ✅ Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/visiondetect-pro.git
   cd visiondetect-pro
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download YOLO model:
   - Place your YOLOv8 `.pt` file in the project directory.
   - Update the `model_path` variable in the code if necessary.

---

## ▶ Usage

### Start the Application
```bash
python visiondetect_pro.py
```

### Configure Settings
- Select your camera from the **Camera Settings** tab.
- Adjust resolution, FPS, and camera properties.
- Load your YOLO model if not automatically loaded.
- Click **Start Inspection** to begin detection.

---

## ⚙ Configuration

### Camera Settings
- **Resolution**: Adjust based on performance needs.
- **FPS Target**: Set desired frames per second.
- **Camera Properties**: Fine-tune exposure, gain, etc.

### Model Settings
- **Confidence Threshold**: Control detection sensitivity (0.1–0.9).
- **IoU Threshold**: Adjust non-maximum suppression (0.1–0.9).
- **Max Detections**: Limit simultaneous detections (1–1000).
- **Class Filtering**: Focus detection on specific classes.

### Output Settings
- Enable or disable frame capture.
- Set save interval and image quality.
- Enable CSV logging of detections.

---

## 📈 Performance Tips

### For Higher FPS:
- Lower the resolution.
- Enable frame skipping.
- Use half-precision (FP16) with GPU acceleration.

### For Better Accuracy:
- Adjust exposure and gain for optimal lighting.
- Fine-tune confidence and IoU thresholds.
- Filter relevant classes.

### For Stability:
- Close other apps using the camera.
- Ensure proper lighting.
- Verify camera drivers and backend support.

---

## 📂 Supported Models

VisionDetect Pro supports:
- YOLOv8 `.pt` format models.
- Object detection, instance segmentation (partial), pose estimation (partial).
- Custom YOLO models trained with Ultralytics YOLO.

---

## 🛠 Troubleshooting

### Common Issues
- **Camera not detected**: Check permissions, drivers, and other applications.
- **Low FPS**: Reduce resolution or enable frame skipping.
- **Model not loading**: Verify file path and compatibility.
- **Detection accuracy issues**: Adjust thresholds and camera settings.

### Getting Help
- Review status messages and console logs.
- Ensure all dependencies are installed.
- Create an issue on the GitHub repository.

---

## 🤝 Contributing

We welcome contributions! Steps to contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request.

Follow PEP 8 guidelines and document your code.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

## 📜 Version History

**2.0.0 (Current)**
- Multi-threaded processing.
- Enhanced camera controls.
- Performance optimizations.
- Settings import/export.

**1.0.0**
- Initial release.
- Basic detection and camera controls.
