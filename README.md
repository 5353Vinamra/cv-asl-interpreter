# ASL Translator — AMD DirectML Edition v5.0

A highly optimized, dual-threaded computer vision pipeline designed to translate American Sign Language (ASL) fingerspelling into text in real time[cite: 3]. This system isolates heavy machine learning tasks from the user interface, achieving a smooth ~60 FPS display loop alongside a dedicated ~12–15 FPS inference worker[cite: 3].

It is highly optimized for hardware acceleration using ONNX Runtime with the DirectML Execution Provider (specifically tuned for AMD hardware like the Radeon 760M, with automatic CPU fallback)[cite: 3].

---

## 🧠 Core Architecture & Pipeline

The translator processes incoming video frames through a sophisticated three-stage verification pipeline to ensure maximum accuracy and eliminate common spatial errors[cite: 3]:

### Stage 1: Gesture Recogniser (Highest Priority)
Evaluates pure landmark geometry to detect 7 universal conflict-free gestures[cite: 3].
*   Detects common phrases like Thumbs Up/Down, I Love You (ILY), Vulcan Salute, Rock On, and Open Palm[cite: 3].
*   If a confident match is found, the system skips subsequent AI stages for that frame, saving significant GPU cycles and preventing YOLO misclassifications[cite: 3].

### Stage 1.5: Motion Letter Detector (J & Z)
Replaces basic coordinate-delta checks with a physics-based velocity and acceleration model[cite: 3].
*   **'J' Detection:** Tracks the pinky tip, looking for an initial downward velocity followed by a leftward hook[cite: 3]. Path curvature is verified via cross-product sign consistency[cite: 3].
*   **'Z' Detection:** Tracks the index tip, looking for a 3-phase velocity profile (+vx → -vx,+vy → +vx) with peak X-acceleration to confirm sharp corners[cite: 3]. 
*   Includes a 10-frame cooldown window to prevent double-firing[cite: 3].

### Stage 2: YOLO ONNX Core
When no static gesture or motion is detected, the system extracts the hand Region of Interest (ROI) and runs it through a YOLO object detection model (`best.onnx`)[cite: 3, 4].
*   **Per-Letter Confidence Thresholds:** Replaces a single adaptive threshold with a dynamic, vectorized NumPy array[cite: 3]. 
*   Hard letters (Q, R, U, X, T) have a lowered threshold to force them into Stage 3 for geometric verification[cite: 3].
*   Easy letters (B, L, V, Y) have a raised threshold to suppress false positives[cite: 3].

### Stage 3: 3D Geo Classifier
Bypasses traditional 2D bounding box limitations by evaluating MediaPipe's 3D `.z` depth coordinates (where negative values are closer to the camera)[cite: 3].
*   **Fist-Cluster 3D Resolver:** Distinguishes between A, S, E, M, N, and T[cite: 3].
*   Calculates the 3D angle of the thumb and checks the signed Z-delta (e.g., if the thumb is in front of or behind the finger PIP joints)[cite: 3].
*   Resolves depth-aware confusion pairs (e.g., K vs P, G vs Q, H vs U)[cite: 3].

---

## 🗣️ Text-to-Speech (TTS) Engine

Includes a dedicated daemon thread running a `pyttsx3` engine (Windows SAPI5)[cite: 3].
*   Operates on a non-blocking queue (max size of 6 utterances) that silently drops stale words to prevent latency[cite: 3].
*   Engine state is safely managed across threads, allowing on-the-fly muting and dynamic stopping/flushing of the TTS queue[cite: 3].

---

## 📁 Repository Structure

*   `main.py`: The core application containing the multithreaded UI, OpenCV loop, MediaPipe landmark extraction, and ONNX inference logic[cite: 3].
*   `best.onnx`: The exported YOLO model weights used for ASL character classification[cite: 3, 4]. *(Note: If this file is over 50MB, it may need to be downloaded separately depending on Git LFS limits).*
*   `hand_landmarker.task`: The MediaPipe task file[cite: 2, 3]. If missing from the directory, `main.py` will automatically download this file at startup[cite: 3].

*(Note: PyTorch checkpoints like `best.pt` and cache files like `main.cpython-313.pyc` are excluded from this repository to maintain a lightweight production environment)[cite: 1, 5].*

---

## ⚙️ Installation & Setup

**1. Prerequisites**
Ensure you have **Python 3.13** (or a compatible 3.x version) installed[cite: 3].

**2. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

```

**3. Install Dependencies**
Install the required libraries using pip:

```bash
pip install opencv-python numpy mediapipe onnxruntime-directml autocorrect pyttsx3

```

Note: If you do not have an AMD/DirectML compatible GPU, the system will automatically fall back to the `CPUExecutionProvider`.

**4. Run the Application**

```bash
python main.py

```

---

## 🎮 Controls

* **SPACE**: Manual commit (locks in the current letter without TTS)


* **ENTER**: Finish word (pushes the word through autocorrect and triggers Text-to-Speech)


* **BACKSPACE**: Delete the last character or restore the previous word


* **A**: Toggle Auto-Commit mode (1.5-second hold)


* **M**: Toggle mirror/flip webcam


* **S**: Toggle Text-to-Speech mute


* **C**: Clear the current word


* **ESC**: Clear all text strings


* **G**: Show/Hide the Gesture + Accuracy Guide on-screen


* **Q**: Quit application gracefully



```

```
