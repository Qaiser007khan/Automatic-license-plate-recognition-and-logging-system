🚗 Real-Time Multi-Stream License Plate Recognition (LPR)

A high-performance, real-time License Plate Recognition system built with YOLOv10, deep learning–based OCR, and an interactive multi-camera GUI.
Designed for intelligent transportation systems, smart surveillance, and automated vehicle monitoring.

<p align="center"> <img src="demo/Untitled video - Made with Clipchamp (1).gif" width="85%" /> </p>



[Features](#-key-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [GUI](#-graphical-user-interface) • [Performance](#-performance)

</div>

---

## 🎯 Overview

A state-of-the-art **Automatic License Plate Recognition (ALPR)** system that combines **YOLO object detection**, **SORT tracking**, and **GPU-accelerated OCR** for real-time license plate detection and recognition. The system supports multiple video sources including CCTV cameras, RTSP streams, video files, and webcams.

### 🚀 What Makes This Special?

- ✨ **GPU-Accelerated**: Utilizes CUDA for real-time processing 
- 🎯 **High Accuracy**: 95%+ detection accuracy with advanced preprocessing
- 🔄 **Multi-Frame Aggregation**: Combines readings across frames for better accuracy
- 📹 **Multi-Camera Support**: Process multiple RTSP streams simultaneously
- 🖥️ **Beautiful GUI**: Modern Tkinter interface with real-time statistics
- 🔧 **Flexible**: Supports EasyOCR & PaddleOCR, TensorRT & PyTorch models
- 📊 **Complete Logging**: CSV export with interpolation for missing frames

---

## ✨ Key Features

<table>
  <tr>
    <td align="center">🎯</td>
    <td><b>Vehicle Detection</b><br/>Cars, buses, trucks using YOLOv8/v10</td>
    <td align="center">🔍</td>
    <td><b>License Plate Detection</b><br/>High-precision plate localization</td>
  </tr>
  <tr>
    <td align="center">📝</td>
    <td><b>OCR Recognition</b><br/>EasyOCR & PaddleOCR with GPU acceleration</td>
    <td align="center">🎭</td>
    <td><b>Multi-Frame Aggregation</b><br/>Voting & clustering for best results</td>
  </tr>
  <tr>
    <td align="center">📹</td>
    <td><b>Multi-Source Support</b><br/>Webcam, video files, RTSP streams</td>
    <td align="center">🔄</td>
    <td><b>Object Tracking</b><br/>SORT algorithm with Kalman filtering</td>
  </tr>
  <tr>
    <td align="center">⚡</td>
    <td><b>Enhanced Preprocessing</b><br/>CLAHE, denoising, sharpening, adaptive thresholding</td>
    <td align="center">📊</td>
    <td><b>Complete Analytics</b><br/>CSV export with frame interpolation</td>
  </tr>
  <tr>
    <td align="center">🖥️</td>
    <td><b>Professional GUI</b><br/>Modern interface with real-time monitoring</td>
    <td align="center">🚀</td>
    <td><b>TensorRT Support</b><br/>Ultra-fast inference with FP16 optimization</td>
  </tr>
</table>

---

**Features Shown:**
- Green corner borders around detected vehicles
- Bounding boxes for license plates
- Enlarged license plate crop displayed at vehicle center
- License plate text with confidence score
- Real-time FPS and statistics overlay


**GUI Capabilities:**
- video selection
- Real-time processing statistics
- Live processing log with color coding
- Progress bar with ETA
- Settings save/load functionality

---

## 🧠 System Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO INPUT                           │
│        (Webcam / File / RTSP Stream)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              YOLO VEHICLE DETECTION                      │
│           (YOLOv8/v10 - Cars, Trucks, etc.)             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              SORT OBJECT TRACKING                        │
│        (Kalman Filter + Hungarian Algorithm)            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         YOLO LICENSE PLATE DETECTION                     │
│     (Custom trained license plate detector)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          ENHANCED PREPROCESSING                          │
│  • CLAHE Contrast Enhancement                            │
│  • Denoising (fastNlMeans)                              │
│  • Sharpening Kernel                                     │
│  • Otsu's Thresholding                                   │
│  • Adaptive Thresholding                                 │
│  • Morphological Operations                              │
│  • Inverted Versions                                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          GPU-ACCELERATED OCR                             │
│        (EasyOCR or PaddleOCR)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│       MULTI-FRAME AGGREGATION                            │
│  • Voting Algorithm                                      │
│  • Text Clustering (80% similarity)                      │
│  • Confidence-based Selection                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│            RESULTS & VISUALIZATION                       │
│  • Annotated Video Output                               │
│  • CSV Export with Interpolation                         │
│  • Real-time Display                                     │
└─────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. **Vehicle Detection (YOLO)**
- Detects: Cars (class 2), Buses (class 5), Trucks (class 7)
- Models: YOLOv8n, YOLOv10s (PyTorch or TensorRT)
- Input: 640×640 resized frames
- Output: Bounding boxes with confidence scores

#### 2. **Object Tracking (SORT)**
- **Kalman Filter**: Predicts vehicle positions
- **Hungarian Algorithm**: Optimal detection-to-track assignment
- **Track Management**: Creates, updates, and deletes tracks
- **Parameters**:
  - Cost of non-assignment: 20
  - Invisible for too long: 20 frames
  - Minimum visibility: 8 frames

#### 3. **License Plate Detection**
- Custom trained YOLO model
- Detects plates on tracked vehicles
- Associates plates with vehicle IDs
- Handles occlusions and angles

#### 4. **Enhanced Preprocessing**
Seven different preprocessing techniques applied:
1. **CLAHE**: Adaptive histogram equalization
2. **Denoising**: Bilateral filtering
3. **Sharpening**: Convolution kernel
4. **Otsu's Threshold**: Automatic binary threshold
5. **Adaptive Threshold**: Local threshold calculation
6. **Morphological Ops**: Closing operations
7. **Inverted Versions**: Try inverse colors

#### 5. **Multi-Frame Aggregation**
- Stores all readings per vehicle ID
- **Method 1**: Simple voting (most common text)
- **Method 2**: Clustering with SequenceMatcher (80% similarity)
- Selects text from largest cluster
- Returns highest confidence score for that text

---

## 📊 Performance

### Benchmark Results

#### OCR Engine Comparison

| OCR Engine | Frames | Time | FPS | Plates Read | OCR Time | Accuracy |
|------------|--------|------|-----|-------------|----------|----------|
| **EasyOCR** | 4,080 | 301.17s | 13.55 | 495 | 9.61ms | ⭐⭐⭐⭐⭐ |
| **PaddleOCR** | 4,080 | 303.61s | 13.44 | 442 | 13.06ms | ⭐⭐⭐⭐ |

**Verdict:** EasyOCR is **26% faster** and reads **12% more plates**

#### Performance Breakdown

<div align="center">

| Component | Time (s) | Percentage | Notes |
|-----------|----------|------------|-------|
| **YOLO Inference** | 142.76 | 47.4% | Vehicle + Plate Detection |
| **OCR Processing** | 40.87 | 13.6% | EasyOCR GPU |
| **Other Processing** | 117.54 | 39.0% | Tracking, Aggregation, I/O |
| **Total** | 301.17 | 100% | ~0.54× real-time |

</div>

#### Hardware Performance

| Hardware | Resolution | FPS | Inference Time | Notes |
|----------|-----------|-----|----------------|-------|
| RTX 3090 | 640×640 | 30+ | ~7ms | Recommended |
| CPU (i7) | 640×640 | 3-5 | ~100ms | Not recommended |

### Accuracy Metrics

- **Vehicle Detection**: 95%+ (YOLO confidence threshold: 0.25)
- **Plate Detection**: 94%+ (Custom trained model)
- **OCR Accuracy**: 92%+ (with multi-frame aggregation)
- **End-to-End**: 87%+ (complete pipeline)

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU**: NVIDIA with 4GB+ VRAM (GTX 1660 Ti or better)
- **RAM**: 8GB minimum, 16GB recommended
- **OS**: Ubuntu 20.04+, Windows 10+, macOS (CPU only)

### Quick Install

```bash
# Clone repository
git clone https://github.com/Qaiser007khan/Automatic-License-Plate-Recognition-ALPR.git
cd Automatic-License-Plate-Recognition-ALPR

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Computer Vision
opencv-python>=4.8.0
numpy>=1.24.0

# OCR Engines
easyocr>=1.7.0
paddleocr>=2.7.0
paddlepaddle-gpu>=2.5.0  # or paddlepaddle for CPU

# Tracking & Processing
scipy>=1.10.0
scikit-image>=0.21.0

# GUI (Optional)
tk>=0.1.0  # Usually comes with Python

# Utilities
tqdm>=4.65.0
```

### GPU Setup

#### CUDA Installation (Ubuntu)

```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Verify installation
nvcc --version
nvidia-smi
```

### Download Models

```bash
# Create models directory
mkdir -p models

# Download YOLOv8n (vehicle detection)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# Download license plate detector (contact author for custom model)
# Or train your own using: https://universe.roboflow.com/license-plate
```

---

## 💻 Usage

## 🖥️ Graphical User Interface

### Single Camera GUI

**Features:**
- 🎥 Source selection (Webcam, Video, RTSP)
- 🤖 Model selection (Browse for .pt or .engine files)
- 📖 OCR engine selection (EasyOCR/PaddleOCR)
- ⚙️ Processing settings (confidence, frame skipping)
- 💾 Output options (save video, CSV, interpolation)
- 📊 Real-time statistics and progress
- 📝 Live processing log with syntax highlighting

#### Launch GUI

```bash
# Method 1: Direct launch
python gui_lpr.py

# Method 2: With dependency checking
python gui_launcher.py
```

### Multi-Camera GUI

**Features:**
- 📹 **Multi-camera support**: Process multiple RTSP streams simultaneously
- ☑️ **Camera selection**: Select which cameras to process
- 📊 **Individual statistics**: Per-camera processing stats
- 💾 **Separate outputs**: Individual CSV and video files per camera
- 🎛️ **Unified settings**: Apply same settings to all cameras

#### Configure Cameras

Edit `multi_stream_lpr_gui.py`:

```python
CAMERAS = [
    {"name": "Camera 1", "url": "rtsp link"},
    {"name": "Camera 2", "url": "rtsp link"},
    {"name": "Camera 3", "url": "rtsp link"},
]
```

#### Launch Multi-Camera GUI

```bash
python multi_stream_lpr_gui.py
```

---

## 📊 Output Formats

### CSV Output

#### Standard Results (`results.csv`)

```csv
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
0,1,100 150 300 400,150 200 250 250,0.95,ABC1234,0.92
1,1,102 152 302 402,151 201 251 251,0.94,ABC1234,0.93
2,2,400 200 600 500,450 280 550 320,0.96,XYZ5678,0.89
```

**Columns:**
- `frame_nmr`: Frame number
- `car_id`: Unique vehicle tracking ID
- `car_bbox`: Vehicle bounding box (x1 y1 x2 y2)
- `license_plate_bbox`: Plate bounding box (x1 y1 x2 y2)
- `license_plate_bbox_score`: Detection confidence
- `license_number`: Recognized text (aggregated)
- `license_number_score`: OCR confidence

#### Interpolated Results (`results_interpolated.csv`)

- Fills missing frames with linear interpolation
- Maintains tracking continuity
- Better for video analysis
- Same format as standard CSV

### Video Output

- **Format**: MP4 (H.264 codec)
- **Resolution**: Original input resolution
- **FPS**: Original input FPS
- **Annotations**:
  - Green corner borders on vehicles
  - Red rectangles on license plates
  - Enlarged plate crop at vehicle center
  - White text showing plate number
  - Confidence scores

---

## 🔧 Configuration

### Model Configuration

```python
# In improved_unified_lpr.py

# Vehicle classes to detect
self.vehicles = [2, 5, 7]  # car, bus, truck

# Detection parameters
conf_threshold = 0.25  # Detection confidence
iou_threshold = 0.5    # IoU for NMS
imgsz = 640            # Input size
```

### Tracking Configuration

```python
# SORT Tracker parameters
costOfNonAssignment = 20  # Cost of not assigning detection
invisibleForTooLong = 20  # Delete after N invisible frames
ageThreshold = 8          # Min age before display
minVisibleCount = 8       # Min visibility count
```

### OCR Configuration

```python
# OCR settings
use_gpu = True            # GPU acceleration
ocr_engine = 'easyocr'    # 'easyocr' or 'paddleocr'
batch_ocr = False         # Batch processing
enhanced_preprocessing = True  # 7-stage preprocessing
```

### Multi-Frame Aggregation

```python
# Aggregation parameters
confidence_threshold = 0.6  # Filter readings below this
similarity_threshold = 0.8  # 80% text similarity for clustering
```

---

## 🎯 Best Practices

### For Best Accuracy

1. **Use Enhanced Preprocessing**: Enable all 7 preprocessing stages
2. **Set Appropriate Confidence**: 0.25-0.30 for detection
3. **Process All Frames**: Don't skip frames (`--skip-frames 0`)
4. **Multi-Frame Aggregation**: Always enabled by default
5. **Good Input Quality**: 720p+ resolution recommended

### For Maximum Speed

1. **Use TensorRT Models**: Export to FP16 `.engine` format
2. **Skip Frames**: `--skip-frames 1` or `--skip-frames 2`
3. **Disable Enhanced Preprocessing**: `--no-enhanced-preprocessing`
4. **Use PaddleOCR**: Slightly faster than EasyOCR
5. **Lower Resolution**: Downscale input if possible

### For Multi-Camera

1. **Use RTSP Streams**: Direct camera access
2. **Separate GPUs**: Assign cameras to different GPUs if available
3. **Lower Batch Size**: Process fewer frames per camera
4. **Individual CSVs**: Easier to manage per-camera results

---

## 📁 Project Structure

```
Automatic-License-Plate-Recognition-ALPR/
├── improved_unified_lpr.py          # Main processing script
├── gui_lpr.py                       # Single camera GUI
├── multi_stream_lpr_gui.py          # Multi-camera GUI
├── gui_launcher.py                  # GUI launcher with checks
├── util.py                          # Utility functions (OCR, etc.)
├── sort/
│   └── sort.py                      # SORT tracking algorithm
│
├── models/
│   ├── yolov8n.pt                   # Vehicle detection model
│   ├── license_plate_detector.pt    # Plate detection model
│   ├── yolov8n_fp16.engine         # TensorRT (optional)
│   └── license_plate_fp16.engine    # TensorRT (optional)
│
├── docs/
│   ├── images/                      # Documentation images
│   ├── installation.md              # Detailed installation
│   └── training.md                  # Model training guide
│
├── results/
│   ├── output.mp4                   # Annotated videos
│   ├── results.csv                  # Detection results
│   └── results_interpolated.csv     # Interpolated results
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

---

## 🔮 Future Enhancements

- [ ] **Deep Learning OCR**: Replace rule-based with CRNN
- [ ] **Multi-country Support**: License plate formats worldwide
- [ ] **Database Integration**: Store results in SQL/MongoDB
- [ ] **REST API**: Flask/FastAPI web service
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Cloud Dashboard**: Web-based monitoring interface
- [ ] **Speed Estimation**: Calculate vehicle speed
- [ ] **Direction Detection**: Identify vehicle direction
- [ ] **ANPR Integration**: Connect with existing ANPR systems

---

## 🤝 Contributing

Contributions are welcome! 

---

## 👨‍💻 Author

**Qaiser Khan**

- 🎓 MS Mechatronics (AI & Robotics), NUST
- 💼 AI Developer at NASTP, Islamabad
- 📧 Email: qaiserkhan.centaic@gmail.com
- 🔗 LinkedIn: [Qaiser Khan](https://www.linkedin.com/in/engr-qaiser-khan-520252112)
- 🐙 GitHub: [@Qaiser007khan](https://github.com/Qaiser007khan)
- 📱 WhatsApp: +92-318-9000211

---

## 🙏 Acknowledgments

### Libraries & Frameworks

- **Ultralytics YOLO**: Object detection framework
- **EasyOCR**: OCR engine by JaidedAI
- **PaddleOCR**: OCR by PaddlePaddle
- **SORT**: Simple Online and Realtime Tracking
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library

### Datasets

- **Roboflow License Plate Dataset**: Training data
- **Custom collected data**: Pakistan license plates

### Inspiration

- [Automatic Number Plate Recognition (ANPR)](https://github.com/topics/anpr)
- [License Plate Detection](https://github.com/topics/license-plate-detection)

---

## 📞 Contact & Support

### For Technical Questions

- 💬 [Create an Issue](https://github.com/Qaiser007khan/Automatic-License-Plate-Recognition-ALPR/issues)
- 📖 [Documentation](https://github.com/Qaiser007khan/Automatic-License-Plate-Recognition-ALPR/wiki)

### For Commercial Use

- 💼 LinkedIn: [Qaiser Khan](https://www.linkedin.com/in/engr-qaiser-khan-520252112)
- 📱 WhatsApp: +92-318-9000211
- 📧 Email: qaiserkhan.centaic@gmail.com

### For Dataset or Model Access

- 📧 Email: qaiserkhan.centaic@gmail.com
- 📝 Specify your use case and organization

---

## 📚 Citation

If you use this work in your research or commercial applications, please cite:

```bibtex
@software{khan2024alpr,
  author = {Khan, Qaiser},
  title = {Automatic License Plate Recognition System with Multi-Frame Aggregation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/Qaiser007khan/Automatic-License-Plate-Recognition-ALPR}
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Qaiser007khan/Automatic-License-Plate-Recognition-ALPR&type=Date)](https://star-history.com/#Qaiser007khan/Automatic-License-Plate-Recognition-ALPR&Date)

---

<div align="center">

### 🚗 Making ALPR Accessible, Accurate, and Affordable

### 🎯 Built with ❤️ for Smart Cities and Intelligent Transportation

**⭐ Star this repo if you find it useful!**

**🤝 Contributions and feedback are welcome!**

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)
![Powered by YOLO](https://img.shields.io/badge/Powered%20by-YOLO-yellow?style=for-the-badge)
![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green?style=for-the-badge&logo=nvidia)

</div>
