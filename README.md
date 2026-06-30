# 🤟 Real-Time ASL (American Sign Language) Translator

A high-performance **real-time American Sign Language (ASL) translator** built with **Python**, **MediaPipe**, **YOLO ONNX**, and **ONNX Runtime DirectML**. The application recognizes static ASL letters, dynamic motion letters, and common hand gestures while converting them into text with optional text-to-speech output.

Designed for **low latency**, **high accuracy**, and **GPU acceleration on AMD hardware**.

---

# ✨ Features

## 🔤 Real-Time ASL Letter Recognition
- Recognizes ASL alphabet (A–Z)
- Live webcam inference
- Confidence-based prediction
- Automatic spelling correction for completed words

---

## 🧠 Multi-Stage Recognition Pipeline

The translator uses a **3-stage hybrid recognition system** instead of relying on a single neural network.

### Stage 1 — Gesture Recognition

Detects common hand gestures using MediaPipe landmark geometry.

Supported gestures:

- 🤟 I Love You
- 👍 Thumbs Up
- 👎 Thumbs Down
- 🤘 Rock On
- 🖕 Middle Finger
- 🖖 Vulcan Salute
- 🖐 Open Palm

These gestures bypass the neural network for maximum speed and reliability.

---

### Stage 1.5 — Motion Letter Detection

Static images cannot identify certain ASL letters.

This project detects motion-based letters using velocity and acceleration analysis.

Supported:

- J
- Z

Features:

- Velocity tracking
- Acceleration analysis
- Curvature detection
- Motion cooldown
- Physics-based trajectory recognition

---

### Stage 2 — YOLO ONNX Classifier

The cropped hand image is classified using a YOLO ONNX model.

Features:

- Per-letter confidence thresholds
- Adaptive ROI confidence
- Fast ONNX Runtime inference
- GPU acceleration with DirectML

---

### Stage 3 — 3D Geometry Classifier

Several ASL letters have extremely similar hand shapes.

A custom geometry classifier resolves difficult cases using MediaPipe's 3D landmarks.

Examples:

- A vs S vs E
- M vs N
- K vs P
- G vs Q
- H vs U
- D vs F
- X vs 1
- T

Uses:

- Depth information
- Thumb position
- Finger angles
- Joint coverage
- 3D geometry analysis

---

# 🚀 Performance

Architecture:

- Dual-threaded pipeline
- Separate UI thread
- Separate AI worker thread

Typical performance:

| Component | Speed |
|----------|---------|
| UI | ~60 FPS |
| AI Worker | ~12–15 FPS |
| Webcam | Real-time |
| Latency | Very Low |

---

# 🛠 Technologies Used

- Python
- OpenCV
- MediaPipe Tasks
- ONNX Runtime
- DirectML
- YOLO ONNX
- NumPy
- Pyttsx3
- AutoCorrect

---

# 📂 Project Structure

```
.
├── main.py
├── best.onnx
├── hand_landmarker.task
└── README.md
```

---

# ⚙ Requirements

Install dependencies:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install onnxruntime-directml
pip install autocorrect
pip install pyttsx3
```

---

# ▶ Running the Project

```bash
python main.py
```

Make sure the following files are present:

- `best.onnx`
- `hand_landmarker.task`

---

# ⌨ Controls

| Key | Action |
|------|--------|
| Space | Commit current letter |
| Enter | Complete word |
| Backspace | Delete previous character |
| A | Toggle Auto Commit |
| M | Toggle Mirror Mode |
| S | Toggle Text-to-Speech |
| G | Open Gesture Guide |
| C | Clear Current Word |
| ESC | Clear Entire Sentence |
| Q | Quit |

---

# 🎙 Text-to-Speech

The translator includes an asynchronous text-to-speech system.

Features:

- Thread-safe speech engine
- Queue-based architecture
- Speaks completed words
- Adjustable speech rate
- Mute toggle

---

# 📊 Recognition Improvements

This version introduces several accuracy improvements:

### ✅ Per-Letter Confidence Thresholds

Instead of using one confidence threshold for every letter, each ASL letter has its own optimized threshold.

Benefits:

- Better precision
- Fewer false positives
- Improved difficult-letter recognition

---

### ✅ 3D Fist Cluster Resolver

Accurately separates:

- A
- S
- E
- M
- N
- T

using:

- Thumb angle
- Thumb depth
- Finger coverage
- 3D landmark geometry

---

### ✅ Motion Recognition

Dynamic letters are detected using:

- Velocity
- Acceleration
- Curvature
- Direction changes

instead of simple coordinate tracking.

---

# 🎯 Applications

- Sign language translation
- Accessibility tools
- Human-computer interaction
- Educational projects
- Computer vision research
- AI demonstrations

---

# 📸 User Interface

The application displays:

- Live webcam feed
- Bounding boxes
- Prediction confidence
- Recognition stage
- FPS
- Current sentence
- Gesture labels
- Recognition history
- Auto-commit indicator

---

# 💻 Hardware

Optimized for:

- AMD GPUs (DirectML)
- Integrated GPUs
- Windows PCs

Falls back to CPU execution if GPU acceleration is unavailable.

---

# 🔮 Future Improvements

- Word prediction
- Sentence-level language model
- Multi-hand recognition
- Continuous sentence recognition
- Additional ASL gestures
- Model training improvements
- Cross-platform support
- Mobile deployment

---

# 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

# 📄 License

This project is intended for educational and research purposes.

Choose an appropriate open-source license before public distribution.

---

# ⭐ Acknowledgements

- Google MediaPipe
- ONNX Runtime
- OpenCV
- NumPy
- Python Community

---

## If you found this project useful, consider giving it a ⭐ on GitHub!
