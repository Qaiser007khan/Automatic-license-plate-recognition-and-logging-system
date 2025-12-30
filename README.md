🚗 Real-Time Multi-Stream License Plate Recognition (LPR)

A high-performance, real-time License Plate Recognition system built with YOLOv10, deep learning–based OCR, and an interactive multi-camera GUI.
Designed for intelligent transportation systems, smart surveillance, and automated vehicle monitoring.

<p align="center"> <img src="demo/Untitled video - Made with Clipchamp (1).gif" width="85%" /> </p>
✨ Key Features

🎥 Multi-stream camera support (RTSP / video / webcam)

🧠 YOLOv10-based vehicle & license plate detection

🔎 Dual OCR pipelines

PaddleOCR

EasyOCR (performance comparison included)

🖥️ Interactive GUI dashboard for real-time monitoring

⚡ GPU-accelerated inference (CUDA)

📊 CSV result logging & interpolation

⏱️ Optimized pipeline with FPS tracking and timing breakdown

🔁 Duplicate plate filtering & temporal consistency

🧩 Project Structure
.
├── improved_unified_lpr.py        # Core LPR pipeline (YOLO + OCR + tracking)
├── multi_streram_lpr_gui.py       # Multi-camera GUI application
├── ocr_comparison.txt             # Detailed OCR performance benchmark
├── demo/
│   └── Untitled video - Made with Clipchamp (1).gif
├── outputs/
│   ├── *.csv
│   └── *_interpolated.csv
└── README.md

🧠 OCR Performance Comparison

A real-world benchmark on 4,080 frames using YOLOv10s on CUDA GPU:

🔹 YOLOv10s + PaddleOCR

Average FPS: 13.44

License Plates Read: 442

Avg OCR Time: 13.06 ms / plate

Processing Speed: 0.54× real-time

🔹 YOLOv10s + EasyOCR

Average FPS: 13.55

License Plates Read: 495

Avg OCR Time: 9.61 ms / plate

Processing Speed: 0.54× real-time

📌 Insight: EasyOCR provided faster OCR inference and higher plate recognition count, making it preferable for real-time deployments.
🚀 Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run Single / Unified LPR Pipeline
python improved_unified_lpr.py
3️⃣ Run Multi-Stream GUI
python multi_streram_lpr_gui.py

📈 Output & Results
⭐ Acknowledgment

If you find this project useful, please consider starring ⭐ the repository — it helps support continued research and development.
