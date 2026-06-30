# ASL Translator

A real-time **American Sign Language (ASL) Translator** built with **Python**, **MediaPipe**, and a custom **YOLO ONNX** model. The application recognizes ASL letters, motion-based signs, and common hand gestures in real time, then converts them into text with optional text-to-speech.

Designed for fast and accurate recognition using a multi-stage recognition pipeline.

---

## Features

- Real-time ASL alphabet recognition (A–Z)
- Motion recognition for dynamic letters (J and Z)
- 3D geometric correction for visually similar letters
- Recognition of common hand gestures
- Live confidence display
- Manual and Auto Commit modes
- Automatic spelling correction
- Built-in text-to-speech
- GPU acceleration through ONNX Runtime DirectML when available
- Automatic download of the MediaPipe hand landmark model on first launch

---

## System Requirements

- Windows 10 or Windows 11
- Python 3.11 or newer
- Webcam

> **Note:** This project is currently developed and tested for Windows.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/ASL-Translator.git
cd ASL-Translator
```

Or download the repository as a ZIP file and extract it.

---

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
python asl_translator.py
```

On the first launch, the required MediaPipe hand landmark model will be downloaded automatically.

---

## Controls

| Key | Function |
|------|----------|
| Space | Commit current letter |
| Enter | Complete current word |
| Backspace | Delete last character |
| A | Toggle Auto Commit |
| M | Toggle Mirror Mode |
| S | Toggle Text-to-Speech |
| G | Show Gesture Guide |
| C | Clear current word |
| Esc | Clear all text |
| Q | Quit application |

---

## Project Structure

```
ASL-Translator/
│
├── asl_translator.py
├── best.onnx
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Recognition Pipeline

The translator uses a multi-stage recognition system for improved accuracy.

- **Stage 1:** Geometry-based gesture recognition
- **Stage 2:** Motion detection for dynamic letters (J and Z)
- **Stage 3:** YOLO ONNX letter classification
- **Stage 4:** 3D geometric verification for visually similar letters

This combination improves recognition accuracy while maintaining real-time performance.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- ONNX Runtime DirectML
- NumPy
- pyttsx3
- AutoCorrect

---

## License

This project is licensed under the MIT License.

---

If you find this project useful, consider giving it a ⭐ on GitHub.
