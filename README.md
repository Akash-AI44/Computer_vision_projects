# Computer Vision & Real-Time AI Projects 👁️🤖

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-FF6F00?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, modular collection of **10 hands-on Computer Vision and Deep Learning projects** implemented in Python. This repository covers the entire learning journey—from foundational image processing (filtering, color spaces, webcam streams) to state-of-the-art neural network inference (YOLOv8 object detection & tracking, MediaPipe 21-point hand landmark estimation, ONNX face detection, EasyOCR text recognition, and an intelligent motion-triggered security camera).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Catalog](#-project-catalog)
- [Repository Structure](#-repository-structure)
- [Tech Stack & Pretrained Models](#-tech-stack--pretrained-models)
- [Installation & Setup](#-installation--setup)
- [Quick Start / How to Run](#-quick-start--how-to-run)
- [Detailed Project Breakdown](#-detailed-project-breakdown)
  - [1. Load & Inspect Image](#1-load--inspect-image)
  - [2. Image Filtering & Edge Detection](#2-image-filtering--edge-detection)
  - [3. Live Webcam Feed](#3-live-webcam-feed)
  - [4. Color Detection & Tracking](#4-color-detection--tracking)
  - [5. Deep Learning Face Detection (YuNet ONNX)](#5-deep-learning-face-detection-yunet-onnx)
  - [6. Hand Landmark Tracking (MediaPipe Tasks)](#6-hand-landmark-tracking-mediapipe-tasks)
  - [7. Real-Time Object Detection (YOLOv8)](#7-real-time-object-detection-yolov8)
  - [8. Multi-Object Tracking with Persistent IDs](#8-multi-object-tracking-with-persistent-ids)
  - [9. Optical Character Recognition (EasyOCR)](#9-optical-character-recognition-easyocr)
  - [10. Motion-Triggered Smart Security Camera](#10-motion-triggered-smart-security-camera)
- [GitHub Upload Guide](#-github-upload-guide)
- [Contributing & License](#-contributing--license)

---

## 🚀 Overview

This repository is designed as a practical, production-ready curriculum and reference repository for Computer Vision practitioners and AI enthusiasts. Each project is contained in an independent, clean script with detailed explanations of:
- Frame-by-frame video processing loops.
- Conversion between color spaces (BGR, Grayscale, HSV, RGB).
- Deep learning inference using lightweight ONNX, PyTorch, and TFLite/Task models.
- Optimization techniques (e.g., conditional model inference using frame differencing to save compute).

---

## 📋 Project Catalog

| # | Project Name | File | Key Tech / Model | Input Source | Description |
|---|--------------|------|------------------|--------------|-------------|
| 1 | **Image Loading & Inspection** | `Load_display_image.py` | OpenCV | Image (`Image.jpg`) | Read, inspect dimensions/channels, and display images. |
| 2 | **Image Filtering & Edge Detection** | `Image_Filters.py` | OpenCV, Gaussian Blur, Canny | Image (`Image.jpg`) | Grayscale conversion, noise filtering, and edge detection. |
| 3 | **Live Webcam Feed** | `Live_webcam_feed.py` | OpenCV `VideoCapture` | Live Webcam | Real-time webcam frame acquisition, rendering, and safe release. |
| 4 | **Color Detection & Segmentation** | `color_detection.py` | OpenCV, HSV Color Space | Live Webcam | HSV color mask filtering, contour finding, and target bounding boxes. |
| 5 | **Deep Learning Face Detection** | `Face_detection.py` | OpenCV DNN, YuNet ONNX | Live Webcam | Real-time face detection with 5 facial landmarks and confidence score. |
| 6 | **Hand Landmark Tracking** | `Hand_tracking.py` | MediaPipe Tasks API | Live Webcam | 21 3D hand landmarks tracking with custom skeletal connections. |
| 7 | **Real-Time Object Detection** | `real_time_object_detection.py` | Ultralytics YOLOv8n | Live Webcam | Multi-class real-time object detection across 80 COCO classes. |
| 8 | **Multi-Object Tracking** | `object_tracking.py` | YOLOv8 Tracking Mode | Live Webcam | Cross-frame object tracking with persistent unique identity IDs. |
| 9 | **Optical Character Recognition (OCR)** | `optical_character_recognition.py` | EasyOCR, PyTorch | Image (`sign.jpg`) | Text extraction, bounding polygon localization, and text overlay. |
| 10 | **Motion-Triggered Security Camera** | `Simple_Motion-Triggered_Security_Cam.py` | Frame Differencing + YOLOv8 | Live Webcam | Smart two-stage security system: motion trigger + AI detection + auto-snapshot. |

---

## 📂 Repository Structure

```text
Computer_vision_projects/
│
├── Load_display_image.py                   # Project 1: Image loading & shape inspection
├── Image_Filters.py                        # Project 2: Grayscale, Gaussian blur, Canny edges
├── Live_webcam_feed.py                     # Project 3: Real-time webcam streaming
├── color_detection.py                      # Project 4: HSV-based color segmentation
├── Face_detection.py                       # Project 5: YuNet ONNX deep learning face detection
├── Hand_tracking.py                        # Project 6: MediaPipe Tasks 21-point hand tracking
├── real_time_object_detection.py           # Project 7: Ultralytics YOLOv8 real-time detection
├── object_tracking.py                      # Project 8: YOLOv8 multi-object persistent tracking
├── optical_character_recognition.py        # Project 9: EasyOCR text reading from images
├── Simple_Motion-Triggered_Security_Cam.py # Project 10: Motion-triggered AI security system
│
├── face_detection_yunet.onnx               # Pretrained YuNet ONNX face detection model (~232 KB)
├── hand_landmarker.task                    # MediaPipe hand landmarker task model (~7.8 MB)
├── yolov8n.pt                              # YOLOv8 nano pre-trained weights (~6.5 MB)
├── Image.jpg                               # Sample test image for CV operations
├── sign.jpg                                # Sample street sign image for OCR test
│
├── .gitignore                              # Git ignore rules for virtualenvs, caches & outputs
├── requirements.txt                        # Complete Python dependency specifications
└── README.md                               # Comprehensive project documentation
```

> **Note on Model Files**: All pretrained models (`face_detection_yunet.onnx`, `hand_landmarker.task`, `yolov8n.pt`) are lightweight (< 10 MB each) and safely fit within standard GitHub file size limits without requiring Git LFS.

---

## 🛠️ Tech Stack & Pretrained Models

- **Language**: Python 3.9+ (tested up to 3.12 / 3.14)
- **Computer Vision**: OpenCV (`opencv-python`, `opencv-contrib-python`)
- **Deep Learning**:
  - **Ultralytics YOLOv8** (`yolov8n.pt` - 3.2M parameters, ultra-fast for CPU/GPU)
  - **YuNet Face Detector** (`face_detection_yunet.onnx` - lightweight ONNX model integrated via OpenCV DNN)
  - **Google MediaPipe** (`hand_landmarker.task` - modern Vision Tasks API for hand pose estimation)
  - **EasyOCR** (PyTorch-based CRAFT + ResNet deep learning OCR engine)
- **Scientific Computing**: NumPy

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Akash-AI44/Computer_vision_projects.git
cd Computer_vision_projects
```

### 2. Create and Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚡ Quick Start / How to Run

From the repository root (or project folder), run any script directly using your activated Python environment:

| Script | Command | Interactive Controls |
|---|---|---|
| Image Inspection | `python Load_display_image.py` | Press any key to close |
| Image Filters | `python Image_Filters.py` | Press any key to close |
| Webcam Feed | `python Live_webcam_feed.py` | Press `q` to exit |
| Color Detection | `python color_detection.py` | Press `q` to exit |
| Face Detection | `python Face_detection.py` | Press `q` to exit |
| Hand Landmark Tracking | `python Hand_tracking.py` | Press `q` to exit |
| YOLOv8 Object Detection | `python real_time_object_detection.py` | Press `q` to exit |
| Object Tracking | `python object_tracking.py` | Press `q` to exit |
| EasyOCR | `python optical_character_recognition.py` | Press any key to close |
| Motion Security Cam | `python Simple_Motion-Triggered_Security_Cam.py` | Press `q` to exit |

---

## 🔍 Detailed Project Breakdown

### 1. Load & Inspect Image
- **File**: `cv-projects/Load_display_image.py`
- **Concepts**: OpenCV `imread`, image dimensions, channel depth, high-GUI window control.
- **Workflow**: Reads `Image.jpg`, verifies integrity, prints `(height, width, channels)`, displays the image window, and releases memory on keypress.

### 2. Image Filtering & Edge Detection
- **File**: `cv-projects/Image_Filters.py`
- **Concepts**: Color space conversion (`COLOR_BGR2GRAY`), Gaussian smoothing, Canny edge detection.
- **Workflow**: Demonstrates the classic computer vision preprocessing pipeline:
  $$\text{Original BGR} \xrightarrow{\text{Grayscale}} \text{Single Channel} \xrightarrow{\text{Gaussian Blur (5x5)}} \text{Denoised} \xrightarrow{\text{Canny (50, 150)}} \text{Edges}$$
  Displays all 4 stages side-by-side for visual comparison.

### 3. Live Webcam Feed
- **File**: `cv-projects/Live_webcam_feed.py`
- **Concepts**: Real-time video I/O, device indexing (`cv2.VideoCapture(0)`), infinite streaming loop, graceful hardware release (`cap.release()`).
- **Workflow**: Captures frames continuously from the default system camera and listens for `ord('q')` to terminate.

### 4. Color Detection & Tracking
- **File**: `cv-projects/color_detection.py`
- **Concepts**: Hue-Saturation-Value (HSV) color model, color range masking (`cv2.inRange`), morphological contour detection (`cv2.findContours`), contour area filtering.
- **Workflow**: Isolates target blue hues using lower `[100, 150, 0]` and upper `[140, 255, 255]` thresholds, extracts external contours, filters out noise (area $> 500\text{ px}$), and draws real-time bounding boxes around moving colored items.

### 5. Deep Learning Face Detection (YuNet ONNX)
- **File**: `cv-projects/Face_detection.py`
- **Model**: `face_detection_yunet.onnx`
- **Concepts**: OpenCV DNN module (`cv2.FaceDetectorYN`), dynamic input resolution adjustment, facial bounding boxes, 5-point facial landmark estimation, confidence scoring.
- **Workflow**: Loads the YuNet ONNX model, updates input dimensions dynamically per webcam frame, and draws detected faces with confidence values.

### 6. Hand Landmark Tracking (MediaPipe Tasks)
- **File**: `cv-projects/Hand_tracking.py`
- **Model**: `hand_landmarker.task`
- **Concepts**: Google MediaPipe Vision Tasks API (`HandLandmarker`), 21 3D hand landmarks, coordinate normalization, custom skeleton connection rendering.
- **Workflow**: Converts BGR frames to RGB `mp.Image`, runs the MediaPipe task graph, normalizes coordinates to frame resolution, and renders green landmark nodes with blue connecting skeletal lines in real time.

### 7. Real-Time Object Detection (YOLOv8)
- **File**: `cv-projects/real_time_object_detection.py`
- **Model**: `yolov8n.pt` (COCO pre-trained weights)
- **Concepts**: Convolutional Neural Networks, single-stage object detection, bounding box regression, multi-class classification.
- **Workflow**: Passes live webcam frames to YOLOv8 Nano, outputs bounding boxes, class labels (people, laptops, cell phones, chairs, etc.), and detection confidence metrics.

### 8. Multi-Object Tracking with Persistent IDs
- **File**: `cv-projects/object_tracking.py`
- **Model**: `yolov8n.pt`
- **Concepts**: Multi-object tracking (MOT), Kalman filtering / ByteTrack association, ID persistence across occlusion.
- **Workflow**: Uses `model.track(frame, persist=True)` to track objects across consecutive video frames, assigning a persistent unique numeric ID to each tracked target.

### 9. Optical Character Recognition (EasyOCR)
- **File**: `cv-projects/optical_character_recognition.py`
- **Model**: CRAFT text detection + ResNet recognition
- **Concepts**: Scene text detection, bounding quadrilateral mapping, text recognition, confidence thresholding.
- **Workflow**: Reads `sign.jpg`, runs deep learning OCR, parses detected bounding boxes, overlays recognized text onto the image, and outputs results with confidence scores to the console.

### 10. Motion-Triggered Smart Security Camera
- **File**: `cv-projects/Simple_Motion-Triggered_Security_Cam.py`
- **Model**: Frame Differencing + YOLOv8 Nano
- **Concepts**: Two-phase edge computing architecture, frame differencing (`cv2.absdiff`), binary thresholding, conditional deep learning inference, automated evidence logging.
- **Workflow**:
  1. Computes absolute differences between consecutive blurred grayscale frames.
  2. Counts non-zero changed pixels (`> 5000` threshold indicates motion).
  3. **Conditional Inference**: Only triggers heavy YOLOv8 deep learning when movement is detected, saving CPU/GPU energy.
  4. Automatically saves timestamped snapshots (`motion_YYYYMMDD_HHMMSS.jpg`) as evidence.

---

## 📤 GitHub Upload Guide

Ready to upload this repository to your GitHub account? Follow these quick steps:

### 1. Initialize Git Repository
```bash
git init
```

### 2. Stage and Commit Files
```bash
git add .
git commit -m "feat: initial commit of computer vision & deep learning projects"
```

> **Note**: The included `.gitignore` automatically prevents `venv/`, `__pycache__/`, `.vscode/`, and generated test snapshots (`motion_*.jpg`) from cluttering your repository.

### 3. Link Remote Repository & Push
If you haven't linked the remote repository yet:

```bash
git branch -M main
git remote add origin https://github.com/Akash-AI44/Computer_vision_projects.git
git push -u origin main
```

---

## 🤝 Contributing & License

Contributions, improvements, and feature additions are welcome! Feel free to:
- Open an Issue for bugs, questions, or model upgrade requests.
- Submit a Pull Request with new computer vision algorithms or improved pipelines.

Distributed under the **MIT License**. See `LICENSE` for details.
