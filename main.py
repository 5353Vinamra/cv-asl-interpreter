"""
ASL Translator — AMD DirectML Edition  v5.0
============================================
Architecture : Dual-thread (UI @ ~60 FPS / AI worker @ ~12-15 FPS)
Inference    : onnxruntime-directml → DmlExecutionProvider (AMD Radeon 760M)

PIPELINE (3 stages):
  Stage 1 — GESTURE RECOGNISER  (pure landmark geometry, 7 conflict-free gestures)
  Stage 1.5 — MOTION LETTER DETECTOR  (velocity/acceleration J and Z)
  Stage 2 — YOLO ONNX  (ASL letters A-Z with per-letter confidence thresholds)
  Stage 3 — GEO CLASSIFIER  (3D fist-cluster + depth-aware confusion pairs)

v5.0 — Three accuracy overhauls:

  ① PER-LETTER CONFIDENCE THRESHOLDS
    Replaced single adaptive threshold with a per-letter delta array.
    Hard letters (Q, R, U, X, T, fist-cluster) get a lowered bar so they
    always reach Stage 3 for geometric verification.
    Easy letters (B, L, V, Y) get a raised bar to suppress false positives.
    Fully vectorized with numpy — zero extra CPU cost.

  ② FIST-CLUSTER 3D RESOLVER  (A / S / E / M / N / T)
    Completely replaces the old 2D tips_near_palm heuristic.
    Uses MediaPipe's .z depth coordinate (negative = closer to camera):
      • 3D angle at THUMB_MCP (THUMB_TIP → THUMB_MCP → INDEX_MCP)
        Large (~90°+) = thumb lateral          → A
        Medium (~60°) = thumb wrapping front   → S / T
        Small  (<45°) = thumb deeply tucked    → E / M / N
      • Signed Z-delta: thumb_tip.z − mean(PIP.z)
        Negative = thumb in FRONT of finger PIPs  → S
        Positive = thumb BEHIND finger tips        → E
      • Y-axis coverage count: how many PIP joints is the thumb tip below?
        3 covered → M  |  2 covered → N
      • T: thumb tip X falls between index_MCP X and middle_MCP X
             at approximately index_MCP Y height, tight angle.
    Also adds T to the confusion set (was missing in v4.1).

  ③ VELOCITY + ACCELERATION MOTION LETTER DETECTOR (J and Z)
    Replaces the coordinate-delta trajectory checks with a physics model:
      J — tracks PINKY TIP (correct for ASL-J which draws with the pinky)
          Detects: initial downward velocity phase → leftward hook phase
          Curvature verified via cross-product sign consistency of velocity vectors
          Tortuosity ratio (arc length / displacement) confirms curved path
      Z — tracks INDEX TIP
          Detects: 3-phase velocity profile: +vx → (−vx,+vy) → +vx
          Requires ≥2 X-velocity sign reversals above jitter threshold
          Middle phase must have downward Y component (diagonal stroke)
          Peak X-acceleration confirms sharp corners (not a smooth drift)
      Cooldown window (10 frames) prevents double-firing.
      Both detectors run every frame; motion flag no longer gates them.
      YOLO outputting J is silently downgraded to I (static handshape = I).
      YOLO outputting Z is silently downgraded to 1 (static handshape = 1).

TTS fixes (v5.0 patch):
  • _tts_state dict replaces bool flag — visible across all threads.
  • _stop_event (threading.Event) replaces app_running bool for TTS loop.
  • engine.stop() before each runAndWait() clears stale SAPI5 state.
  • Per-letter speak() calls removed from commit paths; only speak on Enter.
"""

import ast, cv2, os, queue, sys, threading, time, urllib.request
from collections import Counter, deque

import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from autocorrect import Speller

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
ONNX_MODEL_PATH  = "best.onnx"
MEDIAPIPE_TASK   = "hand_landmarker.task"
MEDIAPIPE_URL    = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
YOLO_INPUT_SIZE  = 224
YOLO_CONF_BASE   = 0.38    # base adaptive threshold (ROI-size adjusted)
CONF_FLOOR       = 0.20    # absolute minimum — never drop below this
ROI_PAD          = 40
BUFFER_FRAMES    = 14
HOLD_FRAMES      = 6
AI_INTERVAL_SEC  = 0.07    # raise to 0.10 on potato PCs
AUTO_COMMIT_SEC  = 1.5
MOTION_THRESH    = 14.0    # px/frame — above this the hand is considered moving
TRAJ_HISTORY     = 20      # frames of position history for motion detector
HISTORY_SIZE     = 8
GESTURE_STABLE_N = 4       # frames a gesture must hold in a 6-frame window
MOTION_COOLDOWN  = 10      # frames between repeated J/Z detections

# ── Text-to-Speech ──────────────────────────────────────────────────────────
TTS_RATE      = 175   # words per minute (150-220 is comfortable)
TTS_VOLUME    = 1.0   # 0.0 – 1.0
TTS_QUEUE_MAX = 6     # utterances queued before oldest is dropped

# ── Per-letter confidence delta array ────────────────────────────────────────
# Added to the ROI-adaptive base threshold before filtering YOLO output.
# Negative  → lower the bar: hard letters that must reach Stage 3 geo verification.
# Positive  → raise the bar: visually distinct letters that rarely need correction.
# Zero (default) → standard adaptive threshold.
LETTER_CONF_DELTA = {
    # Hard geometric twins → must reach geo classifier
    "Q": -0.10, "R": -0.09, "X": -0.09, "U": -0.07,
    # Fist cluster → 3D geo classifier  (T is new in v5)
    "A": -0.05, "S": -0.05, "E": -0.05,
    "M": -0.06, "N": -0.06, "T": -0.07,
    # Motion letters → geo / motion detector takes over
    "I": -0.07, "J": -0.06, "Z": -0.06,
    # Palm-flip pairs → depth-aware geo resolver
    "K": -0.05, "P": -0.05, "G": -0.05,
    # Visually distinct → stay strict, suppress false positives
    "B": +0.04, "C": +0.02, "L": +0.04,
    "O": +0.02, "V": +0.04, "W": +0.03, "Y": +0.04,
}

# ══════════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════════════════
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
_frame_lock   = threading.Lock()
_results_lock = threading.Lock()
shared_frame  = None
ai_results    = {
    "box": None, "label": None, "confidence": 0.0,
    "fps": 0, "provider": "Initialising...",
    "moving": False, "hold_count": 0,
    "geo_active": False, "is_gesture": False,
    "gesture_emoji": "", "commit_label": None,
    "stage_tag": "",    # "GEO3D" / "MOTION" / "GEO" / "GESTURE" / ""
}
app_running = True

# ══════════════════════════════════════════════════════════════════════════════
# ONNX SESSION FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def build_ort_session(model_path):
    print(f"INFO  onnxruntime {ort.__version__}")
    for providers, lbl in [
        ([("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"], "DirectML"),
        (["CPUExecutionProvider"], "CPU"),
    ]:
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1 if lbl == "DirectML" else 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            active = sess.get_providers()[0]
            print(f"OK    Provider: {active}")
            return sess, active
        except Exception as e:
            print(f"WARN  {lbl} failed: {e}")
    raise RuntimeError(f"Cannot load '{model_path}'.")

# ══════════════════════════════════════════════════════════════════════════════
# YOLO PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(bgr_crop, size=YOLO_INPUT_SIZE):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    t   = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(t, (2, 0, 1))[np.newaxis])

# ══════════════════════════════════════════════════════════════════════════════
# YOLO POST-PROCESSING  ①  per-letter threshold delta array
# ══════════════════════════════════════════════════════════════════════════════
def build_delta_array(class_names):
    """Pre-build numpy delta array indexed by class index. Called once at startup."""
    return np.array(
        [LETTER_CONF_DELTA.get(n, 0.0) for n in class_names],
        dtype=np.float32
    )

def postprocess_v2(output, base_thresh, class_names, delta_arr):
    """
    Vectorized per-letter threshold filtering.
    Each anchor's top-1 class is tested against:
        effective_thresh = clip(base_thresh + delta_arr[class_idx], CONF_FLOOR, 0.90)
    Returns (label, confidence) or (None, 0.0).
    """
    preds = output[0]                        # (4+C, N)
    cs    = preds[4:]                        # (C, N)
    N     = cs.shape[1]

    bc = np.argmax(cs, axis=0)               # (N,) — best class index per anchor
    bs = cs[bc, np.arange(N)]               # (N,) — best score per anchor

    # Compute effective per-anchor threshold (vectorized)
    bc_safe    = np.clip(bc, 0, len(delta_arr) - 1)
    eff_thresh = np.clip(base_thresh + delta_arr[bc_safe], CONF_FLOOR, 0.90)

    # Apply per-anchor threshold
    mask = bs >= eff_thresh
    if not mask.any():
        return None, 0.0

    # Best among those that passed
    bs_masked = np.where(mask, bs, -1.0)
    top       = int(np.argmax(bs_masked))
    cls_idx   = int(bc[top])
    if cls_idx >= len(class_names):
        return None, 0.0
    return class_names[cls_idx], float(bs[top])

def adaptive_thresh(roi_area, frame_area):
    ratio = roi_area / max(frame_area, 1)
    return float(np.clip(YOLO_CONF_BASE + (ratio - 0.15) * 0.4, 0.32, 0.55))

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
def lms_to_pts(lms, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

def lms_to_3d(lms):
    """Extract normalized 3D coords (x,y,z) — z is depth relative to wrist."""
    return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

def hand_box(pts, fw, fh, pad=ROI_PAD):
    xs, ys = pts[:, 0], pts[:, 1]
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(fw, int(xs.max()) + pad), min(fh, int(ys.max()) + pad))

def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))

def hand_size(pts):
    return dist(pts, WRIST, MIDDLE_MCP) + 1e-6

def angle_deg(pts, a, b, c):
    v1 = pts[a] - pts[b]; v2 = pts[c] - pts[b]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def motion_score(prev, curr):
    if prev is None or curr is None: return 0.0
    return float(np.mean(np.linalg.norm(curr - prev, axis=1)))

def _ext(pts, mcp, tip, t=0.65):
    return dist(pts, mcp, tip) / hand_size(pts) > t

def _curl(pts, mcp, pip, tip, t=0.50):
    return dist(pts, mcp, tip) / hand_size(pts) < t

def idx_up(pts):    return _ext(pts, INDEX_MCP,  INDEX_TIP)
def mid_up(pts):    return _ext(pts, MIDDLE_MCP, MIDDLE_TIP)
def ring_up(pts):   return _ext(pts, RING_MCP,   RING_TIP)
def pink_up(pts):   return _ext(pts, PINKY_MCP,  PINKY_TIP)
def idx_curl(pts):  return _curl(pts, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)
def mid_curl(pts):  return _curl(pts, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
def ring_curl(pts): return _curl(pts, RING_MCP,   RING_PIP,   RING_TIP)
def pink_curl(pts): return _curl(pts, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)
def thumb_ext(pts, t=0.50): return dist(pts, THUMB_MCP, THUMB_TIP) / hand_size(pts) > t

def thumb_points_up(pts):
    hs = hand_size(pts)
    dy = (pts[THUMB_MCP][1] - pts[THUMB_TIP][1]) / hs
    dx = abs(pts[THUMB_TIP][0] - pts[THUMB_MCP][0]) / hs
    return dy > 0.30 and dy > dx

def thumb_points_down(pts):
    return (pts[THUMB_TIP][1] - pts[WRIST][1]) / hand_size(pts) > 0.35

# ══════════════════════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH
# ══════════════════════════════════════════════════════════════════════════════
# Architecture: a single daemon thread owns the pyttsx3 engine (engines are
# NOT thread-safe — never call engine.say() from more than one thread).
# The main loop puts text into _tts_q; the worker drains it in order.
# If the queue is full the OLDEST item is silently dropped so fresh words
# never pile up behind stale ones.
#
# FIX 1: _tts_state dict replaces a bare bool — mutations are visible across
#         all threads without needing 'global' declarations.
# FIX 2: _stop_event (threading.Event) replaces the app_running bool for the
#         TTS loop — .set() is always visible to the daemon thread.
# FIX 3: speak() is only called when a full word is committed (Enter key).
#         Per-letter TTS caused ~150 ms SAPI5 startup overhead per letter.
# FIX 4: engine.stop() before each runAndWait() clears any stale SAPI5 state.

_tts_q      : queue.Queue    = queue.Queue(maxsize=TTS_QUEUE_MAX)
_tts_state  : dict           = {"muted": False}   # FIX 1
_stop_event : threading.Event = threading.Event() # FIX 2


def speak(text: str) -> None:
    """Non-blocking enqueue.  Drops oldest utterance if queue is full."""
    if _tts_state["muted"] or not text:
        return
    if _tts_q.full():
        try:
            _tts_q.get_nowait()   # discard oldest stale utterance
        except queue.Empty:
            pass
    try:
        _tts_q.put_nowait(text)
    except queue.Full:
        pass                      # race — just skip


def _tts_worker() -> None:
    """
    Owns the pyttsx3 engine for its entire lifetime.
    Initialised here (not in main thread) — pyttsx3 SAPI5 driver is
    COM-based and must be created on the thread that will use it.
    """
    try:
        import pyttsx3
    except ImportError:
        print("WARN  pyttsx3 not found — run:  pip install pyttsx3")
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate',   TTS_RATE)
        engine.setProperty('volume', TTS_VOLUME)
        print("OK    TTS engine ready (Windows SAPI5)")
    except Exception as exc:
        print(f"WARN  TTS init failed: {exc}")
        return

    while not _stop_event.is_set() or not _tts_q.empty():   # FIX 2
        try:
            text = _tts_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if text and not _tts_state["muted"]:
                try:
                    engine.stop()   # FIX 4: clear any stale SAPI5 state
                except Exception:
                    pass
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:
            print(f"WARN  TTS speak error: {exc}")
        finally:
            try:
                _tts_q.task_done()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — GESTURE RECOGNISER  (7 conflict-free gestures)
# ══════════════════════════════════════════════════════════════════════════════
class GestureRecogniser:
    PRIORITY = ["ily", "vulcan", "middle_finger", "rock_on",
                "thumbs_up", "thumbs_down", "open_palm"]
    META = {
        "ily":           ("I Love You",         "🤟", "[ILY]"),
        "vulcan":        ("Live Long & Prosper", "🖖", "[VULCAN]"),
        "middle_finger": ("F*** You",            "🖕", "[!!!]"),
        "rock_on":       ("Rock On",             "🤘", "[ROCK]"),
        "thumbs_up":     ("Thumbs Up",           "👍", "[GOOD]"),
        "thumbs_down":   ("Thumbs Down",         "👎", "[BAD]"),
        "open_palm":     ("Stop / High Five",    "🖐", "[STOP]"),
    }

    @staticmethod
    def _ily(pts):
        return (thumb_ext(pts) and idx_up(pts) and mid_curl(pts)
                and ring_curl(pts) and pink_up(pts))

    @staticmethod
    def _vulcan(pts):
        gap = abs(pts[MIDDLE_TIP][0] - pts[RING_TIP][0]) / hand_size(pts)
        return (idx_up(pts) and mid_up(pts) and ring_up(pts)
                and pink_up(pts) and gap > 0.22)

    @staticmethod
    def _middle_finger(pts):
        return (mid_up(pts)
                and not _ext(pts, INDEX_MCP, INDEX_TIP, 0.55)
                and not _ext(pts, RING_MCP,  RING_TIP,  0.55)
                and not _ext(pts, PINKY_MCP, PINKY_TIP, 0.55))

    @staticmethod
    def _rock_on(pts):
        return (idx_up(pts) and mid_curl(pts) and ring_curl(pts)
                and pink_up(pts) and not thumb_ext(pts, t=0.52))

    @staticmethod
    def _thumbs_up(pts):
        return (thumb_points_up(pts) and thumb_ext(pts) and idx_curl(pts)
                and mid_curl(pts) and ring_curl(pts) and pink_curl(pts))

    @staticmethod
    def _thumbs_down(pts):
        return (thumb_points_down(pts) and thumb_ext(pts) and idx_curl(pts)
                and mid_curl(pts) and ring_curl(pts) and pink_curl(pts))

    @staticmethod
    def _open_palm(pts):
        all_up = idx_up(pts) and mid_up(pts) and ring_up(pts) and pink_up(pts)
        spread = abs(pts[INDEX_TIP][0] - pts[PINKY_TIP][0]) / hand_size(pts) > 0.60
        return all_up and thumb_ext(pts) and spread

    _FN = {
        "ily":           _ily.__func__,
        "vulcan":        _vulcan.__func__,
        "middle_finger": _middle_finger.__func__,
        "rock_on":       _rock_on.__func__,
        "thumbs_up":     _thumbs_up.__func__,
        "thumbs_down":   _thumbs_down.__func__,
        "open_palm":     _open_palm.__func__,
    }

    def recognise(self, pts):
        for key in self.PRIORITY:
            try:
                if self._FN[key](pts):
                    lbl, emoji, token = self.META[key]
                    return key, lbl, emoji, token
            except Exception:
                pass
        return None, None, "", ""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1.5 — MOTION LETTER DETECTOR  (velocity + acceleration J / Z)
# ══════════════════════════════════════════════════════════════════════════════
class MotionLetterDetector:
    """
    Tracks pinky tip (J) and index tip (Z) trajectory each frame.
    Detection is based on velocity profiles and acceleration signatures,
    not raw coordinate deltas, so it rejects random hand movement.

    J detection (pinky tip trajectory):
      Phase 1 – predominantly downward velocity   (early_vy > 0 in image coords)
      Phase 2 – predominantly leftward velocity   (late_vx < 0)
      Curvature – cross-product of consecutive velocity vectors is consistent
      Tortuosity – arc length / displacement > 1.4 (curved, not straight)

    Z detection (index tip trajectory):
      Phase 1 – rightward     (+vx)
      Phase 2 – down-left     (−vx, +vy) — the diagonal stroke
      Phase 3 – rightward     (+vx)
      Requires ≥2 sign reversals in vx above jitter threshold
      Middle third must have positive vy (downward diagonal)
      Peak X-acceleration > threshold (sharp corners, not smooth drift)
    """

    def __init__(self):
        self._pink  = deque(maxlen=TRAJ_HISTORY)
        self._idx   = deque(maxlen=TRAJ_HISTORY)
        self._cd    = {"J": 0, "Z": 0}

    def update(self, pts):
        self._pink.append(pts[PINKY_TIP].copy())
        self._idx.append(pts[INDEX_TIP].copy())
        for k in self._cd:
            if self._cd[k] > 0:
                self._cd[k] -= 1

    def reset(self):
        self._pink.clear()
        self._idx.clear()
        self._cd = {"J": 0, "Z": 0}

    @staticmethod
    def _vel_acc(hist, hs):
        """Return (velocity, acceleration) arrays, normalized by hand size."""
        traj = np.array(hist, dtype=np.float32) / hs
        vel  = np.diff(traj, axis=0)
        acc  = np.diff(vel,  axis=0)
        return vel, acc

    def detect_J(self, pts):
        if self._cd["J"] > 0:                                  return None
        if len(self._pink) < 12:                               return None
        if not (pink_up(pts) and idx_curl(pts)
                and mid_curl(pts) and ring_curl(pts)):          return None

        hs = hand_size(pts)
        vel, acc = self._vel_acc(self._pink, hs)

        path_len = float(np.sum(np.linalg.norm(vel, axis=1)))
        if path_len < 0.15:                                     return None

        half     = max(len(vel) // 2, 1)
        early_vy = float(np.mean(vel[:half, 1]))
        late_vx  = float(np.mean(vel[half:, 0]))

        crosses = np.array([
            vel[i, 0] * vel[i+1, 1] - vel[i, 1] * vel[i+1, 0]
            for i in range(len(vel) - 1)
        ], dtype=np.float32)
        curvature_snr = (float(abs(np.mean(crosses)))
                         / (float(np.std(crosses)) + 1e-6))

        displacement = float(np.linalg.norm(
            np.array(self._pink[-1]) - np.array(self._pink[0])
        )) / hs
        tortuosity = path_len / max(displacement, 0.01)

        if (early_vy > 0.010
                and late_vx < -0.006
                and path_len > 0.15
                and curvature_snr > 0.25
                and tortuosity > 1.40):
            self._cd["J"] = MOTION_COOLDOWN
            return "J"
        return None

    def detect_Z(self, pts):
        if self._cd["Z"] > 0:                                  return None
        if len(self._idx) < 14:                                return None
        if not (idx_up(pts) and mid_curl(pts)
                and ring_curl(pts) and pink_curl(pts)):         return None

        hs = hand_size(pts)
        vel, acc = self._vel_acc(self._idx, hs)

        path_len = float(np.sum(np.linalg.norm(vel, axis=1)))
        if path_len < 0.20:                                     return None

        vx      = vel[:, 0]
        JITTER  = 0.008
        sig_vx  = vx[np.abs(vx) > JITTER]
        if len(sig_vx) < 5:                                    return None
        sign_changes = int(np.sum(np.abs(np.diff(np.sign(sig_vx))) > 0))
        if sign_changes < 2:                                    return None

        third  = max(len(vel) // 3, 1)
        mid_vy = float(np.mean(vel[third:2*third, 1]))
        if mid_vy < 0.004:                                     return None

        max_acc_x = (float(np.max(np.abs(acc[:, 0])))
                     if len(acc) > 0 else 0.0)
        if max_acc_x < 0.003:                                  return None

        if sign_changes >= 2 and mid_vy > 0.004 and path_len > 0.20:
            self._cd["Z"] = MOTION_COOLDOWN
            return "Z"
        return None

    def detect(self, pts):
        j = self.detect_J(pts)
        if j: return j
        return self.detect_Z(pts)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — GEO CLASSIFIER  (3D fist-cluster + updated confusion pairs)
# ══════════════════════════════════════════════════════════════════════════════
class GeoClassifier:
    CONFUSION = frozenset("GQKHUPMNASEDFRXT1I")

    @staticmethod
    def _fist_cluster_3d(p3d, hs3d):
        """
        Resolves A / S / E / M / N / T from the 3D landmark array p3d (21,3).
        hs3d = 3D wrist-to-middle-MCP distance (hand scale).
        """
        for mcp, tip in [(INDEX_MCP,  INDEX_TIP),  (MIDDLE_MCP, MIDDLE_TIP),
                         (RING_MCP,   RING_TIP),   (PINKY_MCP,  PINKY_TIP)]:
            if np.linalg.norm(p3d[mcp] - p3d[tip]) / hs3d > 0.65:
                return None

        v_tip = p3d[THUMB_TIP] - p3d[THUMB_MCP]
        v_idx = p3d[INDEX_MCP] - p3d[THUMB_MCP]
        cos_a = (np.dot(v_tip, v_idx)
                 / (np.linalg.norm(v_tip) * np.linalg.norm(v_idx) + 1e-9))
        thumb_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        avg_pip_z      = float(np.mean([p3d[INDEX_PIP,  2],
                                        p3d[MIDDLE_PIP, 2],
                                        p3d[RING_PIP,   2]]))
        thumb_vs_pip_z = float(p3d[THUMB_TIP, 2] - avg_pip_z)

        ty        = float(p3d[THUMB_TIP, 1])
        covered_i = ty > float(p3d[INDEX_PIP,  1])
        covered_m = ty > float(p3d[MIDDLE_PIP, 1])
        covered_r = ty > float(p3d[RING_PIP,   1])
        n_covered = int(covered_i) + int(covered_m) + int(covered_r)

        palm_y    = float(np.mean([p3d[INDEX_MCP, 1],  p3d[MIDDLE_MCP, 1],
                                   p3d[RING_MCP,  1],  p3d[PINKY_MCP,  1]]))
        tip_ys    = [float(p3d[t, 1]) for t in
                     [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]
        deep_curl = sum(ty_i > palm_y - 0.02 for ty_i in tip_ys)

        idx_x = float(p3d[INDEX_MCP,  0])
        mid_x = float(p3d[MIDDLE_MCP, 0])
        thb_x = float(p3d[THUMB_TIP,  0])
        thb_y = float(p3d[THUMB_TIP,  1])
        idx_y = float(p3d[INDEX_MCP,  1])
        in_x  = min(idx_x, mid_x) - 0.04 < thb_x < max(idx_x, mid_x) + 0.04
        at_y  = abs(thb_y - idx_y) / hs3d < 0.18

        if in_x and at_y and thumb_angle < 55:
            return "T"
        if deep_curl >= 4 and thumb_vs_pip_z >= -0.010:
            return "E"
        if n_covered >= 3:
            return "M"
        if n_covered == 2:
            return "N"
        if thumb_vs_pip_z < -0.018 and thumb_angle < 80:
            return "S"
        if thumb_angle > 70 and n_covered <= 1:
            return "A"
        if n_covered >= 2 and thumb_angle < 65:
            return "M" if n_covered >= 3 else "N"
        return None

    @staticmethod
    def _kp(lms3d):
        return "P" if lms3d[MIDDLE_MCP].z - lms3d[WRIST].z > 0.06 else "K"

    @staticmethod
    def _gq(pts):
        return "Q" if (pts[INDEX_TIP][1] - pts[WRIST][1]) / hand_size(pts) > 0.25 else "G"

    @staticmethod
    def _hu(pts):
        v = pts[INDEX_TIP] - pts[INDEX_MCP]
        return "H" if abs(v[0]) > abs(v[1]) else "U"

    @staticmethod
    def _df(pts):
        hs = hand_size(pts)
        if dist(pts, INDEX_MCP, INDEX_TIP) / hs > 0.85: return "D"
        if dist(pts, THUMB_TIP, INDEX_TIP) / hs < 0.25: return "F"
        return None

    @staticmethod
    def _r(pts):
        hs   = hand_size(pts)
        iext = dist(pts, INDEX_MCP,  INDEX_TIP)  / hs > 0.70
        mext = dist(pts, MIDDLE_MCP, MIDDLE_TIP) / hs > 0.70
        gap  = abs(pts[INDEX_TIP][0] - pts[MIDDLE_TIP][0]) / hs
        return "R" if (iext and mext and gap < 0.38) else None

    @staticmethod
    def _x1(pts):
        return "X" if angle_deg(pts, INDEX_PIP, INDEX_DIP, INDEX_TIP) < 150 else "1"

    def classify(self, yolo_label, pts, lms3d, p3d, hs3d):
        if yolo_label == "J": return "I", True, "GEO"
        if yolo_label == "Z": return "1", True, "GEO"
        if yolo_label not in self.CONFUSION:
            return yolo_label, False, ""

        geo = None
        tag = "GEO"

        if yolo_label in ("A","S","E","M","N","T"):
            geo = self._fist_cluster_3d(p3d, hs3d)
            tag = "GEO3D"
        elif yolo_label in ("K","P"):     geo = self._kp(lms3d)
        elif yolo_label in ("G","Q"):     geo = self._gq(pts)
        elif yolo_label in ("H","U"):     geo = self._hu(pts)
        elif yolo_label in ("D","F"):     geo = self._df(pts)
        elif yolo_label == "R":           geo = self._r(pts)
        elif yolo_label in ("X","1"):     geo = self._x1(pts)
        elif yolo_label == "I":
            return "I", False, ""

        if geo and geo != yolo_label:
            return geo, True, tag
        return yolo_label, False, ""


# ══════════════════════════════════════════════════════════════════════════════
# AI WORKER
# ══════════════════════════════════════════════════════════════════════════════
def ai_worker():
    global shared_frame, ai_results, app_running

    if not os.path.exists(MEDIAPIPE_TASK):
        print("INFO  Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_TASK)
        print("OK    Downloaded.")

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_TASK),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    session, provider = build_ort_session(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    meta = session.get_modelmeta().custom_metadata_map
    if "names" in meta:
        raw = ast.literal_eval(meta["names"])
        class_names = [raw[i] for i in range(len(raw))]
    else:
        class_names = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

    delta_arr = build_delta_array(class_names)

    with _results_lock:
        ai_results["provider"] = provider.replace("ExecutionProvider", "")

    gesture_rec   = GestureRecogniser()
    geo_cls       = GeoClassifier()
    motion_det    = MotionLetterDetector()

    g_vote_buf: deque = deque(maxlen=6)
    vote_window: deque = deque(maxlen=BUFFER_FRAMES)
    hold_label, hold_count = None, 0
    prev_pts  = None
    last_time = time.time()

    print("OK    AI worker live — v5.0 (3D fist cluster + velocity J/Z + per-letter thresh)")

    while app_running:
        loop_start = time.time()

        with _frame_lock:
            frame = shared_frame.copy() if shared_frame is not None else None
        if frame is None:
            time.sleep(0.01); continue

        fh, fw  = frame.shape[:2]
        rgb_f   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_f)
        mp_res  = detector.detect(mp_img)

        best_label    = None
        best_conf     = 0.0
        best_box      = None
        commit_label  = None
        is_moving = geo_active = is_gesture = False
        gesture_emoji = stage_tag = ""

        if mp_res.hand_landmarks:
            lms3d = mp_res.hand_landmarks[0]
            pts   = lms_to_pts(lms3d, fw, fh)
            p3d   = lms_to_3d(lms3d)
            hs3d  = float(np.linalg.norm(p3d[WRIST] - p3d[MIDDLE_MCP])) + 1e-8

            motion    = motion_score(prev_pts, pts)
            is_moving = motion > MOTION_THRESH
            prev_pts  = pts

            motion_det.update(pts)
            motion_letter = motion_det.detect(pts)

            # ── Stage 1: Gesture recogniser ───────────────────────────────
            g_key, g_lbl, g_emoji, g_token = gesture_rec.recognise(pts)
            g_vote_buf.append(g_key)
            top_g = Counter(k for k in g_vote_buf if k is not None)
            stable_key = None
            if top_g:
                ck, cnt = top_g.most_common(1)[0]
                if cnt >= GESTURE_STABLE_N:
                    stable_key = ck

            if stable_key:
                lbl_s, emoji_s, token_s = GestureRecogniser.META[stable_key]
                best_label    = lbl_s
                gesture_emoji = emoji_s
                commit_label  = token_s
                best_conf     = 1.0
                is_gesture    = True
                stage_tag     = "GESTURE"
                best_box      = hand_box(pts, fw, fh)
                vote_window.clear()
                hold_label = token_s
                hold_count = min(hold_count + 1, HOLD_FRAMES + 2)

            # ── Stage 1.5: Motion letter (J / Z) ─────────────────────────
            elif motion_letter:
                best_label   = motion_letter
                commit_label = motion_letter
                best_conf    = 0.88
                geo_active   = True
                stage_tag    = "MOTION"
                best_box     = hand_box(pts, fw, fh)
                vote_window.append((motion_letter, 0.88 ** 2))
                tally = {}
                for l, w in vote_window:
                    tally[l] = tally.get(l, 0.0) + w
                best_label = max(tally, key=tally.get)
                hold_label, hold_count = (
                    (best_label, min(hold_count + 1, HOLD_FRAMES + 2))
                    if best_label == hold_label else (best_label, 1)
                )

            elif not is_moving:
                # ── Stage 2: YOLO with per-letter thresholds ─────────────
                bx   = hand_box(pts, fw, fh)
                crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
                if crop.size > 0:
                    best_box    = bx
                    roi_area    = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    base_thresh = adaptive_thresh(roi_area, fh * fw)
                    tensor      = preprocess(crop)
                    outputs     = session.run(None, {input_name: tensor})
                    raw_lbl, raw_conf = postprocess_v2(
                        outputs[0], base_thresh, class_names, delta_arr)

                    if raw_lbl:
                        # ── Stage 3: Geo correction ────────────────────
                        final_lbl, geo_active, stage_tag = geo_cls.classify(
                            raw_lbl, pts, lms3d, p3d, hs3d)

                        vote_window.append((final_lbl, raw_conf ** 2))
                        tally = {}
                        for l, w in vote_window:
                            tally[l] = tally.get(l, 0.0) + w
                        best_label   = max(tally, key=tally.get)
                        tw           = sum(tally.values())
                        best_conf    = tally[best_label] / tw if tw else raw_conf
                        commit_label = best_label

                        if best_label == hold_label:
                            hold_count = min(hold_count + 1, HOLD_FRAMES + 2)
                        else:
                            hold_label = best_label; hold_count = 1
                    else:
                        vote_window.clear()
                        hold_label, hold_count = None, 0
            else:
                vote_window.clear()
                g_vote_buf.clear()
                hold_label, hold_count = None, 0
        else:
            prev_pts = None
            motion_det.reset()
            vote_window.clear()
            g_vote_buf.clear()
            hold_label, hold_count = None, 0

        curr_time = time.time()
        with _results_lock:
            ai_results.update({
                "box": best_box, "label": best_label,
                "commit_label": commit_label, "confidence": best_conf,
                "fps": int(1 / max(curr_time - last_time, 1e-6)),
                "moving": is_moving, "hold_count": hold_count,
                "geo_active": geo_active, "is_gesture": is_gesture,
                "gesture_emoji": gesture_emoji, "stage_tag": stage_tag,
            })
        last_time = curr_time

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, AI_INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def draw_confidence_bar(frame, x, y, w, conf):
    col = (0,200,0) if conf>0.75 else (0,200,220) if conf>0.55 else (0,100,255)
    cv2.rectangle(frame, (x,y), (x+w,y+8), (40,40,40), -1)
    bw = int(w * conf)
    if bw > 0:
        cv2.rectangle(frame, (x,y), (x+bw,y+8), col, -1)
    cv2.putText(frame, f"{int(conf*100)}%", (x+w+6,y+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

def draw_hold_ring(frame, cx, cy, hc, max_h, radius=44):
    frac  = min(hc / max(max_h, 1), 1.0)
    angle = int(360 * frac)
    col   = (0,255,180) if frac >= 1.0 else (200,200,0)
    cv2.ellipse(frame, (cx,cy), (radius,radius), -90, 0, 360, (50,50,50), 3)
    if angle > 0:
        cv2.ellipse(frame, (cx,cy), (radius,radius), -90, 0, angle, col, 3)

def draw_history_panel(frame, history, x, y, panel_w=190):
    n = min(len(history), HISTORY_SIZE)
    if n == 0: return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+panel_w, y+n*24+16), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "HISTORY", (x+8,y+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
    for i, word in enumerate(reversed(history[-HISTORY_SIZE:])):
        alpha = max(0.35, 1.0 - i*0.12)
        col   = tuple(int(c*alpha) for c in (220,220,220))
        cv2.putText(frame, word[:22], (x+8, y+28+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

def draw_pill(frame, text, x, y, color=(60,180,60)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(frame, (x-4,y-th-4), (x+tw+4,y+4), color, -1)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

def wrap_text(text, max_chars=38):
    if len(text) <= max_chars: return text, ""
    split = text.rfind(" ", 0, max_chars)
    if split == -1: split = max_chars
    return text[:split], text[split:].strip()

STAGE_COLORS = {
    "GESTURE": (255, 140, 0),
    "MOTION":  (0,   180, 255),
    "GEO3D":   (180, 0,   255),
    "GEO":     (200, 80,  255),
    "":        (0,   220, 220),
}

CHEAT_SHEET_ROWS = [
    ("👍  Fist + thumb strictly UP",      "Thumbs Up"),
    ("👎  Fist + thumb past wrist DOWN",  "Thumbs Down"),
    ("🤟  Thumb + Index + Pinky up",      "I Love You (ILY)"),
    ("🤘  Index + Pinky up, NO thumb",    "Rock On"),
    ("🖕  Middle finger only up",         "F*** You"),
    ("🖖  All 4 fingers up, big M/R gap", "Vulcan Salute"),
    ("🖐  All 5 spread wide + thumb out", "Stop / High Five"),
    ("", ""),
    ("S key: toggle speech on / off (MUTE shown in HUD)", "Text-to-Speech"),
    ("J: draw J in air with PINKY (I handshape)", "Motion — blue tag"),
    ("Z: draw Z in air with INDEX (1 handshape)", "Motion — blue tag"),
    ("A/S/E/M/N/T: depth-resolved via 3D",       "Geo3D — violet tag"),
]

def draw_cheat_sheet(frame):
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(fw,fh),(10,10,10),-1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.putText(frame, "GESTURE + ACCURACY GUIDE  (G to close)",
                (20,38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,220), 2)
    cv2.putText(frame, "Box colours: orange=gesture  blue=motion J/Z  violet=3D geo  cyan=YOLO",
                (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1)
    for i, (g, m) in enumerate(CHEAT_SHEET_ROWS):
        y = 88 + i * 36
        cv2.putText(frame, g, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,255,180), 1)
        if m:
            cv2.putText(frame, f"  → {m}", (20, y+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def _quit():
    """Shared teardown: signal all threads then break the camera loop."""
    global app_running
    _stop_event.set()
    app_running = False


print("BOOT  ASL Translator v5.0 — 3D fist cluster + velocity J/Z + per-letter thresh")
if not os.path.exists(ONNX_MODEL_PATH):
    print(f"ERROR  Missing '{ONNX_MODEL_PATH}'.")
    sys.exit(1)

tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="TTS")
tts_thread.start()

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

spell = Speller(lang='en')
cap   = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR  Cannot open webcam.")
    _quit()
    sys.exit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

completed_sentence = ""
current_word       = ""
word_history: list = []
word_has_gesture   = False

auto_commit_mode  = False
auto_commit_start = None
last_auto_label   = None

mirror_mode      = False
show_cheat       = False
last_commit_time = 0.0
COMMIT_DEBOUNCE  = 0.4

print("CAM   Online.  G = guide  |  Q = quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    if mirror_mode:
        frame = cv2.flip(frame, 1)

    with _frame_lock:
        shared_frame = frame.copy()

    fh, fw = frame.shape[:2]

    with _results_lock:
        box           = ai_results["box"]
        label         = ai_results["label"]
        confidence    = ai_results["confidence"]
        ai_fps        = ai_results["fps"]
        provider      = ai_results["provider"]
        is_moving     = ai_results["moving"]
        hold_count    = ai_results["hold_count"]
        geo_active    = ai_results["geo_active"]
        is_gesture    = ai_results["is_gesture"]
        gesture_emoji = ai_results["gesture_emoji"]
        commit_label  = ai_results["commit_label"]
        stage_tag     = ai_results["stage_tag"]

    # ── Cheat-sheet overlay ───────────────────────────────────────────────
    if show_cheat:
        draw_cheat_sheet(frame)
        cv2.imshow("ASL Translator v5.0", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('g'), 27):
            show_cheat = False
        elif key == ord('q'):
            _quit()
            break
        continue

    # ── Auto-commit timing ────────────────────────────────────────────────
    if auto_commit_mode and commit_label:
        if commit_label != last_auto_label:
            last_auto_label = commit_label
            auto_commit_start = time.time()
        elif auto_commit_start and (time.time() - auto_commit_start) >= AUTO_COMMIT_SEC:
            now = time.time()
            if now - last_commit_time > COMMIT_DEBOUNCE:
                current_word += commit_label
                if is_gesture:
                    word_has_gesture = True
                last_commit_time = now
                # FIX 3: no per-letter speak() here
            auto_commit_start = None
            last_auto_label   = None
    elif not auto_commit_mode:
        last_auto_label   = None
        auto_commit_start = None

    auto_frac = 0.0
    if auto_commit_mode and auto_commit_start and label:
        auto_frac = min((time.time() - auto_commit_start) / AUTO_COMMIT_SEC, 1.0)

    # ── DRAW ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, fh-110), (fw, fh), (12,12,12), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if box:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        box_col = STAGE_COLORS.get(stage_tag, (0,220,220))
        if hold_count >= HOLD_FRAMES and not is_gesture:
            box_col = (0,200,60)
        cv2.rectangle(frame, (x1,y1), (x2,y2), box_col, 2)

        if label:
            if is_gesture:
                cv2.putText(frame, gesture_emoji + "  " + label,
                            (x1, max(y1-14, 44)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255,180,0), 3)
                draw_pill(frame, "GESTURE", x1, max(y1-52, 20), (200,100,0))
                cv2.putText(frame, f"→ {commit_label}",
                            (x1, y2+26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,160,80), 1)
            else:
                cv2.putText(frame, label,
                            (x1, max(y1-14, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,80), 3)
                if stage_tag:
                    tag_col = STAGE_COLORS.get(stage_tag, (100,100,100))
                    draw_pill(frame, stage_tag, x1, max(y1-44, 16), tag_col)
                draw_confidence_bar(frame, x1, y2+6, bw, confidence)
                if not auto_commit_mode:
                    cx = x2 + 55
                    cy = (y1 + y2) // 2
                    if cx < fw - 10:
                        draw_hold_ring(frame, cx, cy, hold_count, HOLD_FRAMES)

        if auto_commit_mode and auto_frac > 0 and not is_gesture:
            cx = (x1 + x2) // 2
            cy = y1 - 50
            if cy > 50:
                ang = int(360 * auto_frac)
                cv2.ellipse(frame, (cx,cy), (22,22), -90, 0, 360, (40,40,40), 3)
                cv2.ellipse(frame, (cx,cy), (22,22), -90, 0, ang,  (0,255,150), 3)

    if is_moving and stage_tag != "MOTION":
        draw_pill(frame, "MOVING — HOLD STILL", 10, 45, (0,100,200))

    # HUD — uses _tts_state["muted"] (FIX 1)
    mode_str = "AUTO" if auto_commit_mode else "MANUAL"
    flip_str = " | MIRROR" if mirror_mode else ""
    mute_str = " | MUTE"   if _tts_state["muted"] else ""
    cv2.putText(frame,
                f"AI:{ai_fps}FPS | {provider} | {mode_str}{flip_str}{mute_str} | G=guide",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,220), 1)
    cv2.putText(frame,
                "SPC=commit  ENT=word  A=auto  M=mirror  S=mute  C=clr  ESC=clr all  Q=quit",
                (10, fh-118), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100,100,100), 1)

    display_text = completed_sentence + current_word
    line1, line2 = wrap_text(display_text if display_text else "...")
    cv2.putText(frame, ">" + line1, (12, fh-78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    if line2:
        cv2.putText(frame, " " + line2, (12, fh-44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

    if word_history:
        draw_history_panel(frame, word_history, fw-200, 30)

    cv2.imshow("ASL Translator v5.0", frame)

    # ── KEY HANDLER ───────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        _quit()
        break

    elif key == 27:   # ESC — clear all text
        completed_sentence = ""
        current_word       = ""
        word_has_gesture   = False

    elif key == ord('g'):
        show_cheat = True

    elif key == ord('c'):
        current_word     = ""
        word_has_gesture = False

    elif key == 32:   # SPACE — manual commit (no per-letter TTS, FIX 3)
        if not auto_commit_mode and commit_label:
            now = time.time()
            if now - last_commit_time > COMMIT_DEBOUNCE:
                current_word += commit_label
                if is_gesture:
                    word_has_gesture = True
                last_commit_time = now

    elif key == 13:   # ENTER — finish word, speak it
        if current_word:
            corrected = current_word if word_has_gesture else spell(current_word)
            completed_sentence += corrected + " "
            word_history.append(corrected)
            speak(corrected)   # only TTS call — speaks the completed word
            current_word     = ""
            word_has_gesture = False

    elif key == 8:    # BACKSPACE
        if current_word:
            current_word = current_word[:-1]
        elif completed_sentence:
            parts = completed_sentence.rstrip().rsplit(" ", 1)
            if len(parts) == 2:
                completed_sentence = parts[0] + " "
                current_word       = parts[1]
            else:
                current_word       = parts[0]
                completed_sentence = ""
            if word_history:
                word_history.pop()

    elif key == ord('a'):
        auto_commit_mode  = not auto_commit_mode
        auto_commit_start = None
        last_auto_label   = None
        print(f"INFO  Auto-commit: {'ON' if auto_commit_mode else 'OFF'}")

    elif key == ord('m'):
        mirror_mode = not mirror_mode
        print(f"INFO  Mirror: {'ON' if mirror_mode else 'OFF'}")

    elif key == ord('s'):   # S — toggle TTS mute (FIX 1)
        _tts_state["muted"] = not _tts_state["muted"]
        status = "MUTED" if _tts_state["muted"] else "ON"
        print(f"INFO  TTS: {status}")
        speak("muted" if _tts_state["muted"] else "speech on")

# ── TEARDOWN ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
_quit()               # ensure _stop_event is set even on abnormal exit
ai_thread.join(timeout=2.0)
print("EXIT  Done.")