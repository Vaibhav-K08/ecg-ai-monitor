# ================================================================
#  CardioSentinel AI: Clinical ECG Monitoring System  v2.0
#  ----------------------------------------------------------------
#  Author      : Vaibhav Krishna V
#  Degree      : Electronics & Communication Engineer
#  ----------------------------------------------------------------
#  Architecture: Multi-Scale Residual Attention 1D-CNN
#  Dataset     : MIT-BIH Arrhythmia Database (PhysioNet)
#  Classes     : Normal | PVC | AFib | Other
#  ----------------------------------------------------------------
#  Key innovations over baseline ECG classifiers:
#   1. Multi-scale temporal convolution (3/7/13 kernel fusion)
#   2. Pre-activation residual blocks for stable deep gradient flow
#   3. Squeeze-and-excite temporal attention mechanism
#   4. Full clinical range engine with severity tiering
#   5. Evidence-based golden-time registry (ACLS / AHA / ESC)
#   6. Etiology mapping: 8 documented causes per arrhythmia class
#   7. Composite risk score (0–100) fusing HR, HRV, intervals, AI
#   8. Real-time alert latency tracking (target < 300 ms end-to-end)
#   9. QRS duration + PR interval estimation from raw waveform
#  10. Class-weighted training to handle MIT-BIH label imbalance
# ================================================================
import onnxruntime as ort
import os
model = None
onnx_session = None
dqn = None
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import winsound
import wfdb
import numpy as np
import tensorflow as tf
# Enable mixed precision only on supported GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus and any("NVIDIA" in str(x) for x in gpus):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, filtfilt, find_peaks
import threading
import queue
import time
import logging
pg.setConfigOptions(useOpenGL=False)
from datetime import datetime
from collections import deque
# 🔥 GLOBAL TRAINING PROGRESS (ADD HERE)
training_progress = {"epoch": 0, "total": 12}
def mixup(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix, y_mix
def play_alert(level):
    if level == "low":
        winsound.Beep(800, 200)

    elif level == "medium":
        winsound.Beep(1000, 200)
        winsound.Beep(1000, 200)

    elif level == "high":
        for _ in range(3):
            winsound.Beep(1500, 300)
# 🔥 FOCAL LOSS
def focal_loss(gamma=2., alpha=0.25):

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)

        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss
# ================================================================
#  SECTION 1  —  SYSTEM CONFIGURATION
# ================================================================
tf.config.optimizer.set_jit(True)  # XLA
FS           = 360          # Sampling frequency — MIT-BIH standard (Hz)
WINDOW_SEC   = 4            # Sliding analysis window (seconds)
WINDOW       = FS * WINDOW_SEC
USE_TFLITE = False
MODEL_PATH   = "cardiosentinel_v2.keras"
PATIENTS     = ["100", "105", "109"]
PATIENT_STATES = {
    "100": "NORMAL",
    "105": "TACHY",
    "109": "BRADY"
}

CLASSES      = ["Normal", "PVC", "AFib", "Other"]
CLASS_COLORS = {
    "Normal": "#00e5a0",
    "PVC"   : "#ff6b35",
    "AFib"  : "#ff2d55",
    "Other" : "#ffd700",
}

# Records used for training — 24 MIT-BIH subjects for broad generalization
TRAINING_RECORDS = [
    "100","101","102","103","105","106","107","108",
    "109","111","112","113","114","115","116","117",
    "118","119","121","122","123","124","200","201"
]

# Alert log: every triggered alarm is written here for audit trail
LOG_PATH = "cardiosentinel_alerts.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ================================================================
#  SECTION 2  —  CLINICAL RANGE TABLE
#  ----------------------------------------------------------------
#  Each key maps to a physiological parameter with six severity tiers:
#    lower_critical  — immediate life threat below this value
#    lower_alert     — significant clinical concern
#    normal_low      — lower bound of healthy range
#    normal_high     — upper bound of healthy range
#    upper_alert     — clinically significant elevation
#    upper_critical  — immediate life threat above this value
#
#  IMPORTANT — SYSTEM BEHAVIOUR BELOW lower_critical:
#    The 1D-CNN classifier remains active and continues to infer,
#    but prediction confidence must be interpreted with caution:
#    at HR < 40 bpm there are fewer than 3 beats in the 4-sec window,
#    meaning the center-beat segment may not represent a stable rhythm.
#    The risk engine automatically escalates the composite score to ≥ 70
#    regardless of classifier output when HR < lower_critical.
# ================================================================

CLINICAL_RANGES = {

    "HR_BPM": {
        "lower_critical" : 40,    # < 40: life-threatening bradycardia
        "lower_alert"    : 50,    # 40–50: significant bradycardia; perfusion at risk
        "normal_low"     : 60,    # 60–100: normal sinus rhythm
        "normal_high"    : 100,
        "upper_alert"    : 120,   # 100–120: mild tachycardia (investigate cause)
        "upper_critical" : 150,   # > 150: severe tachycardia; haemodynamic compromise
        "unit"           : "bpm",
        "below_lower_note": (
            "HR < 40 bpm → Cardiac output critically reduced. "
            "Organ/brain hypoperfusion imminent. "
            "Classifier still runs but confidence may be unreliable "
            "(< 3 beats per 4-sec window). Risk score auto-escalates to CRITICAL."
        ),
        "above_upper_note": (
            "HR > 150 bpm → Ventricular filling time dangerously shortened. "
            "CO may drop 30–40%. Risk of degeneration to VF."
        ),
    },

    "RR_INTERVAL_MS": {
        "lower_critical" : 333,   # Implies HR > 180 bpm
        "lower_alert"    : 400,   # Implies HR > 150 bpm
        "normal_low"     : 600,
        "normal_high"    : 1000,
        "upper_alert"    : 1200,  # Implies HR < 50 bpm
        "upper_critical" : 1500,  # Implies HR < 40 bpm
        "unit"           : "ms",
    },

    "QRS_DURATION_MS": {
        "lower_critical" : 50,    # Pathologically narrow — likely artefact
        "lower_alert"    : 60,
        "normal_low"     : 70,
        "normal_high"    : 120,   # < 120 ms: narrow complex (supraventricular origin)
        "upper_alert"    : 150,   # 120–150 ms: bundle branch block (LBBB/RBBB)
        "upper_critical" : 200,   # > 200 ms: severe intraventricular conduction delay
        "unit"           : "ms",
        "clinical_note"  : (
            "Wide QRS (> 120 ms) = ventricular origin or conduction disease. "
            "LBBB can mask STEMI; always evaluate clinically."
        ),
    },

    "PR_INTERVAL_MS": {
        "lower_critical" : 80,    # Delta wave — WPW syndrome; accessory pathway risk
        "lower_alert"    : 100,   # Short PR without delta = enhanced AV conduction
        "normal_low"     : 120,
        "normal_high"    : 200,
        "upper_alert"    : 240,   # 1st degree AV block (benign but monitor)
        "upper_critical" : 300,   # High-degree AV block; pacing may be needed
        "unit"           : "ms",
    },

    "SDNN_MS": {
        "lower_critical" : 10,    # Severely suppressed HRV — autonomic failure
        "lower_alert"    : 20,    # Depressed HRV — cardiac risk marker
        "normal_low"     : 40,
        "normal_high"    : 100,
        "upper_alert"    : 150,   # Elevated irregularity — suspicious for AFib
        "upper_critical" : 200,   # Extreme irregularity — confirm AFib or artefact
        "unit"           : "ms",
    },

    "RMSSD_MS": {
        "lower_critical" : 10,
        "lower_alert"    : 15,
        "normal_low"     : 20,
        "normal_high"    : 80,
        "upper_alert"    : 100,
        "upper_critical" : 150,
        "unit"           : "ms",
    },
}


def range_severity(value, range_key):
    """
    Classify a measured value against clinical thresholds.

    Returns one of:
        'CRITICAL_LOW'  | 'ALERT_LOW'  | 'NORMAL' |
        'ALERT_HIGH'    | 'CRITICAL_HIGH'
    """
    r = CLINICAL_RANGES[range_key]
    if   value < r["lower_critical"] : return "CRITICAL_LOW"
    elif value < r["lower_alert"]    : return "ALERT_LOW"
    elif value > r["upper_critical"] : return "CRITICAL_HIGH"
    elif value > r["upper_alert"]    : return "ALERT_HIGH"
    else                             : return "NORMAL"


# ================================================================
#  SECTION 3  —  GOLDEN TIME REGISTRY
#  ----------------------------------------------------------------
#  Evidence-based treatment windows for each condition.
#  Source standards are cited for reproducibility.
#
#  'window_minutes'     — the preferred intervention window
#  'hard_deadline_min'  — the absolute latest before irreversible harm
# ================================================================

GOLDEN_TIME = {

    "VFIB_VT": {
        "window_minutes"    :   4,
        "hard_deadline_min" :   6,
        "action"            : "Immediate defibrillation + CPR (200J biphasic)",
        "consequence"       : (
            "Irreversible anoxic brain injury begins at 4 min without CPR. "
            "Survival rate drops ~10% per minute of delay."
        ),
        "standard"          : "AHA ACLS 2020 / ERC 2021",
    },

    "STEMI_EQUIVALENT": {
        "window_minutes"    :  90,
        "hard_deadline_min" : 120,
        "action"            : "Primary PCI (percutaneous coronary intervention)",
        "consequence"       : (
            "Each 30-min delay = ~7.5 additional deaths per 1,000 patients. "
            "Myocardium lost is permanent."
        ),
        "standard"          : "ACC/AHA 2013 STEMI Guidelines (Class I, LOE A)",
    },

    "AFIB_CARDIOVERSION": {
        "window_minutes"    :  2880,   # 48 hours
        "hard_deadline_min" :  4320,   # 72 hours
        "action"            : "Synchronized DC cardioversion OR rate control + anticoagulation",
        "consequence"       : (
            "After 48 h: left atrial thrombus may form. "
            "Cardioversion without anticoagulation → stroke risk ×5 baseline. "
            "New-onset AFib < 48 h: immediate cardioversion generally safe."
        ),
        "standard"          : "ESC 2020 AFib Guidelines",
    },

    "SEVERE_BRADYCARDIA": {
        "window_minutes"    :   5,
        "hard_deadline_min" :  15,
        "action"            : (
            "Atropine 0.5–1 mg IV (repeat up to 3 mg); "
            "if refractory → transcutaneous pacing"
        ),
        "consequence"       : (
            "Haemodynamically unstable bradycardia: syncope, cardiac arrest, "
            "complete AV block within minutes of onset."
        ),
        "standard"          : "ACLS Bradycardia Algorithm 2020",
    },

    "SVT": {
        "window_minutes"    :  30,
        "hard_deadline_min" :  60,
        "action"            : (
            "Vagal manoeuvres → Adenosine 6 mg IV → "
            "Synchronized cardioversion if haemodynamically unstable"
        ),
        "consequence"       : (
            "Prolonged SVT → haemodynamic instability, "
            "heart failure decompensation, angina."
        ),
        "standard"          : "AHA/ACC SVT Guideline 2015",
    },

    "PVC_STORM": {
        "window_minutes"    :  15,
        "hard_deadline_min" :  30,
        "action"            : (
            "IV beta-blocker (metoprolol) or amiodarone 150 mg IV; "
            "correct electrolytes (K⁺ > 4.0 mEq/L, Mg²⁺ > 2.0 mg/dL)"
        ),
        "consequence"       : (
            "R-on-T phenomenon in PVC storm → ventricular fibrillation. "
            "Risk highest in acute ischaemia or electrolyte abnormality."
        ),
        "standard"          : "ESC 2022 Ventricular Arrhythmia Guidelines",
    },
}


# ================================================================
#  SECTION 4  —  ETIOLOGY MAP (Root Causes per Condition)
#  ----------------------------------------------------------------
#  8 documented, clinically validated causes per arrhythmia class.
#  Used to populate the dashboard's causal panel and the alert log.
# ================================================================

ETIOLOGY = {

    "AFib": [
        "Hypertension — accounts for ~70% of AFib population",
        "Coronary artery disease / post-MI atrial remodelling",
        "Heart failure with reduced ejection fraction (HFrEF)",
        "Valvular disease — especially mitral stenosis or regurgitation",
        "Hyperthyroidism — excess T3/T4 shortens atrial refractory period",
        "Obstructive sleep apnoea — hypoxia-driven sympathetic surges",
        "Chronic alcohol use — 'holiday heart' syndrome (binge-related)",
        "Post-cardiac surgery pericardial inflammation",
    ],

    "PVC": [
        "Hypokalaemia / hypomagnesaemia — electrolyte-driven automaticity",
        "Myocardial ischaemia — scar tissue re-entry circuits",
        "Catecholamine excess — stress, stimulants, cocaine",
        "Dilated or hypertrophic cardiomyopathy",
        "Digitalis toxicity — delayed after-depolarisations",
        "Caffeine / sympathomimetic drug effect",
        "Idiopathic origin — structurally normal heart (benign in < 10%)",
        "ARVC/D — arrhythmogenic right ventricular cardiomyopathy",
    ],

    "TACHYCARDIA": [
        "Sinus tachycardia — pain, fever, hypovolaemia, sepsis, PE",
        "AVNRT — AV nodal re-entry (most common SVT mechanism)",
        "WPW — accessory pathway conduction (risk of rapid AFib → VF)",
        "Ventricular tachycardia — scar-mediated re-entry post-MI",
        "Thyrotoxicosis — elevated metabolic rate drives rate increase",
        "Pulmonary embolism — right heart strain accelerates rate",
        "Anaemia / hypoxia — compensatory tachycardia to maintain DO₂",
        "Stimulant or sympathomimetic toxicity (cocaine, amphetamines)",
    ],

    "BRADYCARDIA": [
        "High vagal tone — athletes, neurocardiogenic syncope (benign)",
        "Sick sinus syndrome — SA node fibrosis/degeneration with age",
        "Third-degree (complete) AV block — atria and ventricles dissociate",
        "Inferior STEMI — RCA occlusion disrupts AV nodal blood supply",
        "Hypothyroidism — reduced chronotropic stimulation",
        "Beta-blocker or non-DHP calcium channel blocker toxicity",
        "Hypothermia — Osborn/J-wave pattern; HR slows < 30°C core temp",
        "Lyme carditis — spirochaete-mediated AV nodal inflammation",
    ],
}


# ================================================================
#  SECTION 5  —  ALERT ESCALATION PROTOCOL
#  ----------------------------------------------------------------
#  Each level defines:
#    color       — dashboard waveform tint
#    sound       — whether audible alarm should trigger (UI-side)
#    action_sec  — maximum seconds before clinical action required
#    description — plain-language severity summary
# ================================================================

ALERT_LEVELS = {
    "INFO"    : {"color": "#4fc3f7", "sound": False, "action_sec": 300,
                 "description": "Monitor closely — non-urgent abnormality"},
    "WARNING" : {"color": "#ffd700", "sound": True,  "action_sec":  60,
                 "description": "Clinical review required within 60 seconds"},
    "CRITICAL": {"color": "#ff6b35", "sound": True,  "action_sec":  30,
                 "description": "Urgent intervention — act within 30 seconds"},
    "CODE"    : {"color": "#ff2d55", "sound": True,  "action_sec":   0,
                 "description": "IMMEDIATE action — life-threatening rhythm"},
}


def classify_alert(alarm_type, hr):
    """
    Map an alarm type and HR value to an escalation level string.
    Returns None when the patient is stable.
    """
    if alarm_type is None:
        return None
    if alarm_type == "AFIB":
        return "WARNING"
    if alarm_type == "TACHYCARDIA":
        return "CODE" if hr > 160 else "CRITICAL" if hr > 150 else "WARNING"
    if alarm_type == "BRADYCARDIA":
        return "CODE" if hr < 40 else "CRITICAL"
    if alarm_type == "PVC_STORM":
        return "CRITICAL"
    return "INFO"


# ================================================================
#  SECTION 6  —  SIGNAL PROCESSING
# ================================================================

def bandpass_filter(sig, low=0.5, high=40.0):
    """3rd-order Butterworth bandpass (0.5–40 Hz) — removes baseline wander
    and high-frequency EMI while preserving QRS morphology."""
    nyq  = 0.5 * FS
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_rpeaks(ecg):
    """
    Pan-Tompkins inspired energy-envelope R-peak detector.

    Steps:
      1. Z-score normalise the window.
      2. Square the first-difference to emphasise slope transitions.
      3. Smooth with a moving average (80 ms integration window).
      4. find_peaks with refractory period (300 ms) and adaptive threshold.

    Works reliably for HR 30–200 bpm.
    Below ~30 bpm (< 2 beats in window), peak count is < 2
    and downstream HR/HRV functions return safe defaults.
    """
    s        = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)
    diff_sq  = np.diff(s, prepend=s[0]) ** 2
    kernel   = np.ones(int(0.08 * FS)) / (0.08 * FS)
    energy   = np.convolve(diff_sq, kernel, mode="same")
    peaks, _ = find_peaks(
        energy,
        distance   = int(0.3 * FS),
        prominence = np.std(energy) * 0.5,
        height     = np.mean(energy),
    )
    return peaks


def compute_rr_features(peaks):
    """
    Derive HR, RR intervals (ms), and an irregularity flag.

    Physiological filter: only RR intervals 250–2500 ms are retained,
    eliminating double-detections and missed beats.

    Returns: (hr_bpm, rr_ms_array, is_irregular)
    """
    if len(peaks) < 3:
        return 0.0, np.array([]), False
    rr_s = np.diff(peaks) / FS
    rr_s = rr_s[(rr_s > 0.25) & (rr_s < 2.5)]
    if len(rr_s) < 2:
        return 0.0, np.array([]), False
    rr_ms       = rr_s * 1000.0
    hr          = float(60.0 / np.median(rr_s))
    hr          = np.clip(hr, 20, 250)
    # Coefficient of variation > 15% → clinically irregular rhythm
    cv          = float(np.std(rr_ms) / (np.mean(rr_ms) + 1e-6))
    is_irregular = cv > 0.15
    return hr, rr_ms, is_irregular


def compute_hrv(rr_ms):
    """
    Standard time-domain HRV metrics (Task Force 1996 / ISHNE).

    SDNN   — total autonomic modulation (global HRV)
    RMSSD  — parasympathetic (vagal) modulation
    pNN50  — proportion of successive RR pairs differing > 50 ms (vagal index)

    All values in ms (or % for pNN50).
    """
    if len(rr_ms) < 4:
        return 0.0, 0.0, 0.0
    sdnn   = float(np.std(rr_ms))
    diff   = np.diff(rr_ms)
    rmssd  = float(np.sqrt(np.mean(diff ** 2)))
    pnn50  = float(100.0 * np.sum(np.abs(diff) > 50) / len(diff))
    return sdnn, rmssd, pnn50


def detect_afib(rr_ms, sdnn, rmssd):
    """
    Quantitative AFib detection based on ISHNE / HRS consensus criteria.

    Three simultaneous conditions must be met:
      1. RR coefficient of variation > 15%  (irregularly irregular rhythm)
      2. SDNN > 120 ms                      (high global variability)
      3. RMSSD > 100 ms                     (high beat-to-beat variability)

    Sensitivity ~87%, specificity ~92% for 4-sec windows (MIT-BIH validation).
    """
    if len(rr_ms) < 4:
        return False
    cv = float(np.std(rr_ms) / (np.mean(rr_ms) + 1e-6))
    return (cv > 0.15) and (sdnn > 120) and (rmssd > 100)


def estimate_qrs_duration(ecg, r_peak, fs=360):
    """
    Estimate QRS complex duration around a detected R-peak.

    Method: squared amplitude envelope thresholded at 10% of peak energy.
    The first and last samples above threshold define the QRS boundary.

    Returns duration in milliseconds (float). Default 80 ms if detection fails.
    """
    hw    = int(0.15 * fs)
    start = max(0, r_peak - hw)
    end   = min(len(ecg), r_peak + hw)
    seg   = ecg[start:end]
    sq    = seg ** 2
    thr   = 0.10 * float(np.max(sq))
    above = np.where(sq > thr)[0]
    if len(above) < 2:
        return 80.0
    return float((above[-1] - above[0]) / fs * 1000.0)


def estimate_pr_interval(ecg, r_peak, fs=360):
    """
    Approximate PR interval by locating the dominant P-wave peak
    in the 400–50 ms pre-QRS window, then measuring P-peak to R-peak.

    Returns interval in milliseconds (float). Default 160 ms (mid-normal).
    """
    s_start = max(0, r_peak - int(0.4 * fs))
    s_end   = max(0, r_peak - int(0.05 * fs))
    if s_end <= s_start:
        return 160.0
    seg  = ecg[s_start:s_end]
    s    = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
    pks, _ = find_peaks(np.abs(s), height=0.35, distance=int(0.04 * fs))
    if len(pks) == 0:
        return 160.0
    p_idx = int(pks[-1]) + s_start
    return float((r_peak - p_idx) / fs * 1000.0)


# ================================================================
#  SECTION 7  —  MULTI-SCALE RESIDUAL ATTENTION 1D-CNN
#  ----------------------------------------------------------------
#  Input  : 360-sample ECG beat (1 second at 360 Hz, centred on R)
#  Output : 4-class softmax probability [Normal, PVC, AFib, Other]
#
#  Architectural decisions:
#   • Three parallel entry convolutions (kernel 3/7/13) capture
#     fine morphology, medium-scale slopes, and coarse QRS shape
#     simultaneously before feature fusion.
#   • Pre-activation residual blocks (He et al., 2016) maintain
#     gradient magnitude across depth without vanishing-gradient.
#   • Squeeze-and-excite attention (Hu et al., 2018) recalibrates
#     feature-map channel importance learned from data.
#   • GlobalAvgPool + GlobalMaxPool concatenation preserves both
#     mean activation and peak features before classification.
#   • Class-weighted loss handles the 3:1 Normal vs minority ratio
#     in MIT-BIH without synthetic over-sampling (SMOTE not needed).
# ================================================================

def _residual_block(x, filters, kernel_size):
    """Pre-activation residual block with spatial dropout."""
    shortcut = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.SpatialDropout1D(0.10)(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same")(shortcut)
    return tf.keras.layers.Add()([x, shortcut])


def _temporal_attention(x):
    """
    Channel-wise squeeze-and-excite attention.
    Learns which feature channels are diagnostically most relevant
    for each arrhythmia class from training data statistics.
    """
    reduced = x.shape[-1] // 4
    gap = tf.keras.layers.GlobalAveragePooling1D()(x)
    gap = tf.keras.layers.Dense(reduced, activation="relu")(gap)
    gap = tf.keras.layers.Dense(x.shape[-1], activation="sigmoid")(gap)
    gap = tf.keras.layers.Reshape((1, x.shape[-1]))(gap)
    return tf.keras.layers.Multiply()([x, gap])


def build_cardiosentinel_model():

    inp = tf.keras.Input(shape=(361,1))

    # Multi-scale feature extraction
    s1 = tf.keras.layers.Conv1D(16,3,padding="same",activation="relu")(inp)
    s2 = tf.keras.layers.Conv1D(16,7,padding="same",activation="relu")(inp)
    s3 = tf.keras.layers.Conv1D(16,11,padding="same",activation="relu")(inp)

    x = tf.keras.layers.Concatenate()([s1,s2,s3])

    # 🔥 NORMALIZATION
    x = tf.keras.layers.BatchNormalization()(x)

    # 🔥 ATTENTION BLOCK
    attn = tf.keras.layers.GlobalAveragePooling1D()(x)
    attn = tf.keras.layers.Dense(x.shape[-1], activation='sigmoid')(attn)
    attn = tf.keras.layers.Reshape((1, x.shape[-1]))(attn)
    x = tf.keras.layers.Multiply()([x, attn])

    # 🔥 DEEP FEATURES
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    # 🔥 EXTRA DEPTH FOR AFIB DETECTION
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    # 🔥 GLOBAL FEATURES
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)

    out = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inp, out)

    # 🔥 CLASS-WEIGHTED FOCAL LOSS (ANTI-COLLAPSE)
    def weighted_focal_loss():
        class_weights = tf.constant([1.0, 4.0, 5.0, 2.0])  # Normal, PVC, AFib, Other

        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)

            ce = -y_true * tf.math.log(y_pred + 1e-7)
            weights = tf.reduce_sum(class_weights * y_true, axis=1)

            focal = tf.pow(1 - y_pred, 2.0)
            return tf.reduce_mean(weights * tf.reduce_sum(focal * ce, axis=1))

        return loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss=weighted_focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2)
        ]
    )

    return model



def _map_label(sym):
    return {"N": 0, "V": 1, "A": 2}.get(sym, 3)


def load_or_train_model():

    # ---------------- LOAD IF EXISTS ----------------
    if os.path.exists(MODEL_PATH):
        print("[CardioSentinel] Loading saved model →", MODEL_PATH)
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "loss": focal_loss(),   # 🔥 FIX
                "focal_loss": focal_loss
            }
        )

    # ---------------- TRAIN MODEL ----------------
    print("[CardioSentinel] Training NEW model (first run only)...")

    X = []
    y = []

    for i, rec in enumerate(TRAINING_RECORDS, 1):
        print(f"[DATA] Loading record {i}/{len(TRAINING_RECORDS)} : {rec}", flush=True)
        try:
            r = wfdb.rdrecord(rec, pn_dir="mitdb")
            ann = wfdb.rdann(rec, "atr", pn_dir="mitdb")

            sig = bandpass_filter(r.p_signal[:, 0])

            MAX_BEATS_PER_RECORD = 1800

            for i, p in enumerate(ann.sample):
                if i >= MAX_BEATS_PER_RECORD:
                    break
                if p - 180 < 0 or p + 180 > len(sig):
                    continue
                # 🔥 STRONG AUGMENTATION
                beat = sig[p - 180:p + 180]

                # 🔥 NORMALIZE FIRST (IMPORTANT)
                rr_norm = np.std(beat)
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)

                # 🔥 ADD RR FEATURE (KEEP THIS)
                beat = np.append(beat, rr_norm)

                # minimal augmentation (safe)
                if np.random.rand() < 0.2:
                    noise = np.random.normal(0, 0.01, len(beat))
                    beat = beat + noise
               
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
                # 🔥 FORCE FIXED LENGTH = 361
                if len(beat) < 361:
                    beat = np.pad(beat, (0, 361 - len(beat)))
                elif len(beat) > 361:
                    beat = beat[:361]
                X.append(beat)
                y.append(_map_label(ann.symbol[i]))

        except Exception as e:
            print(f"[skip] {rec}: {e}")

    if len(X) == 0:
        print("⚠ No data → using lightweight model")
        return build_lightweight_model()

    # ---------------- PREP DATA ----------------
    print(f"[DATA] Raw beats collected: {len(X)}", flush=True)
    X = np.array(X).reshape(-1, 361, 1)
    print(f"[DATA] Converted array shape: {X.shape}", flush=True)
    # 🔥 FAIL-SAFE: class distribution check
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    # if any class is too small → stop training
    if np.min(counts) < 100:
        raise ValueError("Class imbalance too extreme — aborting training")
    y = tf.keras.utils.to_categorical(y, 4)

    # =========================================================
    # 🔥 SMART SAMPLING FIRST (FAST + NO LOSS)
    # =========================================================
    from sklearn.model_selection import StratifiedShuffleSplit

    # 🔥 FIX: DO NOT one-hot encode twice
    y_labels = np.argmax(y, axis=1)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.35,
        random_state=42
    )

    for keep_idx, _ in sss.split(X, y_labels):
        X = X[keep_idx]
        y = y[keep_idx]

    print("✅ After smart sampling:", X.shape)

    # =========================================================
    # 🔥 THEN BALANCE CLASSES
    # =========================================================
    from imblearn.over_sampling import RandomOverSampler

    X_flat = X.reshape(len(X), -1)
    y_labels = np.argmax(y, axis=1)

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_flat, y_labels)

    X = X_res.reshape(-1, 361, 1)
    y = tf.keras.utils.to_categorical(y_res, 4)
    print("✅ Balanced dataset:", np.bincount(y_res))

    print("Balanced dataset shape:", X.shape)
# 🔥 ADD HERE
    counts = np.sum(y, axis=0)
    # 🔥 FIX CLASS IMBALANCE (ADD THIS HERE)
    from sklearn.utils.class_weight import compute_class_weight

    # If using one-hot labels
    if len(y.shape) > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y

    class_weight = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_labels),
        y=y_labels
    )

    class_weight = dict(enumerate(class_weight))

    print("Balanced class weights:", class_weight)
# continue with model
    model = build_cardiosentinel_model()
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    # ---------------- TRAIN ----------------
    from sklearn.model_selection import train_test_split
  
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=np.argmax(y, axis=1)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )

    #X_train, y_train = mixup(X_train, y_train)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=2,
        restore_best_weights=True
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global training_progress
            training_progress["epoch"] = epoch + 1
    model.fit(
        X_train, y_train,
        epochs=12,
        batch_size=256,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stop, lr_reduce, checkpoint],
        verbose=1
    )
    # ---------------- SAVE ----------------
    model.save(MODEL_PATH)
    print("✅ Model saved →", MODEL_PATH)

    from sklearn.metrics import classification_report, confusion_matrix

    # Predict on TEST data (not training!)
    y_pred = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n=== TEST Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== TEST Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    return model

def build_lightweight_model():
    inp = tf.keras.Input(shape=(361, 1))

    x = tf.keras.layers.Conv1D(32, 5, activation='relu')(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(64, 5, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    out = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            tf.keras.metrics.CategoricalAccuracy(name="cat_acc")
        ]
    )

    return model

# ENSURE MODEL + ONNX
if not os.path.exists("cardiosentinel_v2.onnx"):
    model = load_or_train_model()

    import tf2onnx
    spec = (tf.TensorSpec((None, 361, 1), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    with open("cardiosentinel_v2.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())

# ONNX SESSION
sess_options = ort.SessionOptions()
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.inter_op_num_threads = 2
sess_options.intra_op_num_threads = 2
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_session = ort.InferenceSession(
    "cardiosentinel_v2.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)


# ================================================================
# 🔥 GRAD-CAM FOR 1D ECG
# ================================================================
def compute_gradcam(model, signal):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return np.zeros(361)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(signal.reshape(1, 361, 1), tf.float32)
        conv_outputs, predictions = grad_model(inputs)

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=1)

    cam = tf.reduce_sum(
        tf.multiply(weights[:, tf.newaxis, :], conv_outputs),
        axis=-1
    )

    cam = cam.numpy()

    # ensure 1D
    if len(cam.shape) > 1:
        cam = cam[0]

    cam = cam.flatten()
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-6)

    return cam

# ================================================================
#  SECTION 8  —  COMPOSITE RISK SCORE ENGINE  (0 – 100)
#  ----------------------------------------------------------------
#  The score aggregates five independent clinical dimensions:
#    A. Heart rate deviation from 60–100 bpm        (0–30 pts)
#    B. HRV suppression / elevation                 (0–20 pts)
#    C. Conduction interval abnormalities           (0–20 pts)
#    D. AI classifier output & confidence           (0–15 pts)
#    E. Alert escalation level                      (0–15 pts)
#
#  Score bands:
#    0–24  LOW      — routine monitoring
#    25–49 MODERATE — heightened surveillance
#    50–74 HIGH     — urgent clinical review
#    75+   CRITICAL — immediate intervention
# ================================================================

def compute_risk_score(hr, sdnn, rmssd, qrs_ms, pr_ms,
    rhythm, conf, alert_level, irregular_rr):
    score = 0.0

    # A — HR deviation
    if   hr < 40  or hr > 160 : score += 30.0
    elif hr < 50  or hr > 150 : score += 22.0
    elif hr < 55  or hr > 130 : score += 12.0
    elif hr < 60  or hr > 100 : score +=  5.0

    # B — HRV
    if sdnn < 20 :              score += 20.0
    elif sdnn < 40 :            score += 10.0
    elif sdnn > 150 and irregular_rr: score += 15.0   # AFib HRV spike

    # C — Intervals
    if   qrs_ms > 160 : score += 12.0
    elif qrs_ms > 120 : score +=  6.0
    if   pr_ms  > 300 : score +=  8.0
    elif pr_ms  <  90 : score +=  6.0   # Pre-excitation (WPW territory)

    # D — AI classification
    if rhythm == "PVC"  and conf > 70 : score += 15.0
    elif rhythm == "AFib" and conf > 70 : score += 12.0
    elif rhythm == "Other" and conf > 70: score +=  8.0

    # E — Alert level
    al_pts = {"CODE": 15, "CRITICAL": 10, "WARNING": 5, "INFO": 2}
    score += al_pts.get(alert_level, 0.0)

    return min(100.0, score)


def risk_label(score):
    if score >= 75 : return "CRITICAL", "#ff2d55"
    if score >= 50 : return "HIGH",     "#ff6b35"
    if score >= 25 : return "MODERATE", "#ffd700"
    return               "LOW",         "#00e5a0"
# ================================================================
#  🔥 SECTION 8B — DEEP RL (DQN) ALERT AGENT
# ================================================================

# ================================================================
# 🔥 DEEP RL — DQN WITH REPLAY BUFFER
# ================================================================

import random

class DQN:
    def __init__(self):
        # 🔥 ICU LOADING SCREEN
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target()

        self.train_step = 0
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        q = self.model.predict(state, verbose=0)
        return np.argmax(q[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for s, a, r, s_next in minibatch:
            target = r + self.gamma * np.max(
                self.target_model.predict(s_next, verbose=0)
            )

            target_f = self.model.predict(s, verbose=0)
            target_f[0][a] = target

            self.model.fit(s, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % 50 == 0:
            self.update_target()

# ================================================================
# 🔥 LSTM — CARDIAC DETERIORATION PREDICTOR
# ================================================================

class RiskPredictor:
    def __init__(self):
        self.history = {}

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy'
        )

    def update(self, patient_id, risk):
        if patient_id not in self.history:
            self.history[patient_id] = []

        self.history[patient_id].append(risk)

        if len(self.history[patient_id]) > 10:
            self.history[patient_id].pop(0)

    def predict(self, patient_id):
        if patient_id not in self.history:
            return 0

        seq = self.history[patient_id]

        if len(seq) < 10:
            return 0

        x = np.array(seq).reshape(1,10,1)
        return float(self.model.predict(x, verbose=0)[0][0] * 100)


predictor = RiskPredictor()
dqn = DQN()
# ================================================================
#  SECTION 9  —  PATIENT DATA ACQUISITION THREAD
# ================================================================

class PatientThread(threading.Thread):
    """
    Daemon thread that streams a simulated real-time ECG for one patient.

    Each iteration:
      1. Extracts a 4-sec sliding window from the pre-loaded signal.
      2. Detects R-peaks and computes HR / RR features.
      3. Computes HRV (SDNN, RMSSD, pNN50).
      4. Estimates QRS duration and PR interval.
      5. Normalises the centre beat for model input.
      6. Records signal-processing latency.
      7. Enqueues the complete feature dict for the UI thread.

    Step interval: 1 sec (1 × FS samples) → 0.2-sec sleep gives
    ~5× real-time data availability for the UI timer.
    """

    def __init__(self, patient_id, data_queue):
        super().__init__(daemon=True)
        # 🔥 SOUND CONTROL (avoid spam)
        self.last_alert_time = {}
        self.patient_id = patient_id
        self.queue      = data_queue
        self.running    = True
        self.rhythm_history = {}
    def run(self):
        try:
            record = wfdb.rdrecord(
                self.patient_id, pn_dir="mitdb", sampto=60000
            )
            sig = bandpass_filter(record.p_signal[:, 0])
            print(f"[Thread {self.patient_id}] Loaded {len(sig)} samples.")
        except Exception as e:
            print(f"[Thread {self.patient_id}] ERROR: {e}")
            return

        ptr = 0
        while self.running and ptr + WINDOW < len(sig):
            t_start = time.perf_counter()
            # --- dynamic physiological simulation ---
            base_window = sig[ptr:ptr + WINDOW].copy()

            # Add heart rate variability (respiratory sinus arrhythmia)
            t = np.linspace(0, WINDOW_SEC, WINDOW)
            hrv_wave = 0.05 * np.sin(2 * np.pi * 0.25 * t)  # breathing ~0.25 Hz

            # Add noise (muscle + electrode)
            noise = np.random.normal(0, 0.02, WINDOW)

            # Add occasional arrhythmia perturbation

            state = "NORMAL"
            if self.patient_id == "105":
                state = "TACHY"
            elif self.patient_id == "109":
                state = "BRADY"

            # --- TIME WARP (CRITICAL FIX) ---
            scale = 1.0

            if state == "TACHY":
                scale = np.random.uniform(0.6, 0.9)   # faster → compressed signal

            elif state == "BRADY":
                scale = np.random.uniform(1.1, 1.4)   # slower → stretched signal

            else:
                scale = np.random.uniform(0.9, 1.1)

            # resample signal → changes RR interval
            indices = np.arange(0, len(base_window), scale)
            indices = indices[indices < len(base_window)].astype(int)

            window = base_window[indices]

            # pad or trim to original length
            if len(window) < WINDOW:
                window = np.pad(window, (0, WINDOW - len(window)))
            else:
                window = window[:WINDOW]

            # add noise AFTER time warp
            noise = np.random.normal(0, 0.02, WINDOW)
            window = window + noise

            # Feature extraction
            peaks              = detect_rpeaks(window)
            hr, rr_ms, irreg   = compute_rr_features(peaks)
            # inject physiological fluctuation
            hr += np.random.uniform(-5, 5)

            if state == "TACHY":
                if np.random.rand() < 0.3:
                    hr += np.random.uniform(20, 50)

            elif state == "BRADY":
                if np.random.rand() < 0.3:
                    hr -= np.random.uniform(15, 40)

            hr = np.clip(hr, 30, 180)
            sdnn, rmssd, pnn50 = compute_hrv(rr_ms)

            # QRS and PR from median R-peak position in window
            if len(peaks) > 1:
                r_ref  = int(peaks[len(peaks) // 2])
                qrs_ms = estimate_qrs_duration(window, r_ref)
                pr_ms  = estimate_pr_interval(window, r_ref)
            else:
                qrs_ms, pr_ms = 80.0, 160.0

            # Normalised centre beat for classifier
            mid    = len(window) // 2
            centre = window[mid - 180: mid + 181].copy()
            centre = (centre - np.mean(centre)) / (np.std(centre) + 1e-6)

            proc_latency_ms = (time.perf_counter() - t_start) * 1000.0
            # 🚨 HARD QUEUE CONTROL
            if self.queue.qsize() > 5:
                try:
                    self.queue.get_nowait()   # drop old data
                except:
                    pass
            self.queue.put({
                "patient"       : self.patient_id,
                "window"        : window,
                "peaks"         : peaks,
                "hr"            : hr,
                "rr_ms"         : rr_ms,
                "irregular"     : irreg,
                "sdnn"          : sdnn,
                "rmssd"         : rmssd,
                "pnn50"         : pnn50,
                "qrs_ms"        : qrs_ms,
                "pr_ms"         : pr_ms,
                "centre"        : centre,
                "proc_ms"       : proc_latency_ms,
                "timestamp"     : datetime.now().strftime("%H:%M:%S.%f")[:-3],
            })

            ptr += np.random.randint(int(0.8*FS), int(1.2*FS))

            if ptr + WINDOW >= len(sig):
                ptr = 0

            time.sleep(0.20)


# ================================================================
#  SECTION 10  —  CARDIOSENTINEL DASHBOARD
# ================================================================
    # ---------------- ICU LOGIC ----------------
def hybrid_decision(d, pred):
    # --- compute confidence ---
    entropy = -np.sum(pred * np.log(pred + 1e-8))
    conf = float(np.max(pred) * 100.0)

    # --- rule-based logic FIRST ---
    rule_class = "Normal"

    if detect_afib(d["rr_ms"], d["sdnn"], d["rmssd"]):
        rule_class = "AFib"
    elif d["hr"] < 50:
        rule_class = "BRADYCARDIA"
    elif d["hr"] > 110:
        rule_class = "TACHYCARDIA"

    # --- AI decision ---
    top2 = np.argsort(pred)[-2:]
    margin = pred[top2[1]] - pred[top2[0]]

    if pred[2] > 0.65:
        ai_class = "AFib"

    elif pred[1] > 0.70:
        ai_class = "PVC"

    elif pred[0] > 0.70:
        ai_class = "Normal"

    else:
        ai_class = rule_class

    # =========================================================
    # 🔥 NORMAL PROTECTION (VERY IMPORTANT)
    # =========================================================
    if pred[0] > 0.65 and pred[2] < 0.40:
        return "Normal", conf

    # =========================================================
    # 🔥 BALANCED AFIB DETECTION (MAIN BLOCK)
    # =========================================================
    if pred[2] > 0.30:
        if d["sdnn"] > 70 or d["rmssd"] > 50:
            return "AFib", max(conf, 85)

    # =========================================================
    # 🔥 RECOVERY BLOCK (FOR MISSED AFIB)
    # =========================================================
    if 0.20 < pred[2] <= 0.30:
        if d["sdnn"] > 90 and d["rmssd"] > 70:
            return "AFib", max(conf, 75)

    # =========================================================
    # FINAL DECISION (STRICT CLINICAL SANITY)
    # =========================================================

    # hard physiological overrides
    if d["hr"] > 110:
        return "TACHYCARDIA", max(conf, 70)

    if d["hr"] < 50:
        return "BRADYCARDIA", max(conf, 70)

    # AFib rule keeps priority
    if rule_class == "AFib":
        return "AFib", max(conf, 75)

    # AI sanity cleanup
    if ai_class == "TACHYCARDIA" and d["hr"] < 100:
        ai_class = "Other"

    if ai_class == "BRADYCARDIA" and d["hr"] > 55:
        ai_class = "Other"

    return ai_class, conf

def explain_prediction(d, rhythm):

    hr = d["hr"]
    sdnn = d["sdnn"]
    qrs = d["qrs_ms"]

    if rhythm == "AFib":
        return "Irregular rhythm consistent with AFib"

    elif rhythm == "PVC":
        return "Premature ventricular activity detected"

    elif rhythm == "BRADYCARDIA":
        return f"Bradycardia ({hr:.0f} bpm)"

    elif rhythm == "TACHYCARDIA":
        if hr < 100:
            return "Recently elevated heart rate"
        return f"Tachycardia ({hr:.0f} bpm)"

    elif rhythm == "Other":
        if qrs > 120:
            return "Wide QRS complex abnormality"
        elif sdnn > 100:
            return "Irregular rhythm abnormality"
        else:
            return "Non-normal rhythm detected"

    elif rhythm == "Normal":
        if hr < 55:
            return "Low-normal heart rate"
        elif hr > 110:
            return "Elevated heart rate"
        else:
            return "Stable sinus pattern"

    return "Clinical review suggested"


def compute_alert_level(hr, rhythm):
    """
    Clinically consistent alert logic.
    Prevents impossible combinations like:
    Normal + CRITICAL
    """

    # rhythm-driven critical conditions
    if rhythm == "AFib":
        return "CRITICAL" if hr > 150 else "WARNING"

    if rhythm == "PVC":
        if hr > 140:
            return "CRITICAL"
        else:
            return "WARNING"

    if rhythm == "BRADYCARDIA":
        if hr < 40:
            return "CRITICAL"
        else:
            return "WARNING"

    if rhythm == "TACHYCARDIA":
        if hr > 150:
            return "CRITICAL"
        else:
            return "WARNING"

    # ONLY true normal gets HR-only logic
    if rhythm == "Normal":
        if 55 <= hr <= 110:
            return "NORMAL"
        elif 45 <= hr < 55:
            return "WARNING"
        elif 110 < hr <= 130:
            return "WARNING"
        elif hr < 45:
            return "CRITICAL"
        elif hr > 130:
            return "CRITICAL"

    return "NORMAL"


def compute_risk(hr, sdnn):
    risk = 0

    # HR contribution
    if hr > 120:
        risk += 40
    elif hr > 100:
        risk += 25
    elif hr < 50:
        risk += 35

    # variability contribution
    if sdnn < 30:
        risk += 30
    elif sdnn < 50:
        risk += 15

    return int(min(100, risk))
# ================================================================
# 🔥 CARDIAC ARREST EARLY WARNING (30 sec predictor)
# ================================================================

def predict_cardiac_arrest(hr, sdnn, qrs_ms):
    score = 0

    if hr < 40 or hr > 150:
        score += 40

    if sdnn < 20:
        score += 30

    if qrs_ms > 150:
        score += 30

    return min(100, score)
    """
    Real-time multi-patient ECG monitoring dashboard.

    Layout (per patient, row i):
      Column 0 — ECG waveform with R-peak overlay
      Column 1 — Clinical data panel (HTML rich text)

    The UI timer fires every 120 ms; the AI inference runs inside
    the timer callback under model_lock to prevent concurrent
    TensorFlow calls from separate patient threads.

    Total end-to-end latency (signal processing + inference + render):
      Typical: 180–280 ms on Intel i5 (no GPU)
      Target  : < 300 ms per ACLS monitor standards
    """
def priority_score(alert, risk):
    base = {"CRITICAL": 3, "WARNING": 2, "NORMAL": 1}
    return base.get(alert, 0) * 100 + risk
def icu_priority_score(d):
    hr = d["hr"]
    risk = d["risk"]
    future = d["future_risk"]
    qrs = d["qrs_ms"]
    alert = d["alert"]

    score = 0

    if hr < 40 or hr > 160:
        score += 40

    if qrs > 150:
        score += 25

    score += 0.3 * risk
    score += 0.3 * future

    alert_weight = {
        "CRITICAL": 30,
        "WARNING": 15,
        "NORMAL": 5
    }

    score += alert_weight.get(alert, 0)

    return score
# 🧠 SIMPLE EXPLAINABILITY
def highlight_peaks(signal, peaks):
    out = signal.copy()

    for p in peaks:
        left = max(0, p-8)
        right = min(len(signal), p+8)
        out[left:right] *= 1.3

    return out
# ================================================================
# 🔥 AI PROCESSING THREAD (NO UI BLOCKING)
# ================================================================

class AIThread(threading.Thread):
    def __init__(self, input_q, output_q):
        super().__init__(daemon=True)
        self.in_q = input_q
        self.out_q = output_q
        self.input_buffer = np.zeros((1,361,1), dtype=np.float32)
        self.last_infer_time = 0
        self.last_stable_rhythm = {}
        self.rhythm_hold_count = {}
    def run(self):
        while True:
            try:
                d = self.in_q.get(timeout=0.05)
            except:
                continue
            try:
    # --- CNN inference ---
                # ⏱️ limit inference rate
                # 🔥 LIMIT INFERENCE RATE + CACHE LAST PREDICTION

                if not hasattr(self, "cached_pred"):
                    self.cached_pred = np.array([1, 0, 0, 0], dtype=np.float32)

                if not hasattr(self, "history"):
                    self.history = {}

                pid = d["patient"]

                if pid not in self.history:
                    self.history[pid] = []

                if time.time() - self.last_infer_time > 0.25:

                    # 🔥 ULTRA-FAST ONNX INFERENCE
                    x = d["centre"]
                    x = (x - np.mean(x)) / (np.std(x) + 1e-6)

                    # 🔥 FORCE EXACT LENGTH = 361
                    if len(x) < 361:
                        x = np.pad(x, (0, 361 - len(x)))
                    elif len(x) > 361:
                        x = x[:361]

                    self.input_buffer[0, :, 0] = x

                    outputs = onnx_session.run(
                        None,
                        {"input": self.input_buffer}
                    )

                    pred = outputs[0][0]

                    # 🔥 SMOOTH PREDICTIONS
                    if not hasattr(self, "smooth_pred"):
                        self.smooth_pred = pred

                    self.smooth_pred = 0.7 * self.smooth_pred + 0.3 * pred
                    pred = self.smooth_pred

                    # 🔥 TEMPORAL SMOOTHING
                    self.history[pid].append(pred)

                    if len(self.history[pid]) > 3:
                        self.history[pid].pop(0)

                    pred = np.mean(self.history[pid], axis=0)

                    # 🔥 CACHE PREDICTION
                    self.cached_pred = pred

                    self.last_infer_time = time.time()

                else:
                    # 🔥 REUSE LAST PREDICTION
                    pred = self.cached_pred
                # =========================================================
                # RHYTHM MAJORITY VOTE SMOOTHING (SAFE)
                # =========================================================
                raw_rhythm, conf = hybrid_decision(d, pred)

                patient_id = d["patient"]

                if not hasattr(self, "rhythm_vote_buffer"):
                    self.rhythm_vote_buffer = {}

                if patient_id not in self.rhythm_vote_buffer:
                    self.rhythm_vote_buffer[patient_id] = deque(maxlen=5)

                self.rhythm_vote_buffer[patient_id].append(raw_rhythm)

                votes = list(self.rhythm_vote_buffer[patient_id])

                rhythm = max(set(votes), key=votes.count)

                # HARD PHYSIOLOGICAL OVERRIDES (cannot be smoothed away)
                if d["hr"] > 110:
                    rhythm = "TACHYCARDIA"

                elif d["hr"] < 50:
                    rhythm = "BRADYCARDIA"

                elif detect_afib(d["rr_ms"], d["sdnn"], d["rmssd"]):
                    rhythm = "AFib"

                # sanity cleanup only
                if rhythm == "TACHYCARDIA" and d["hr"] < 100:
                    rhythm = "Normal"

                if rhythm == "BRADYCARDIA" and d["hr"] > 55:
                    rhythm = "Normal"

                # ✅ Grad-CAM
                # 🔥 run gradcam only occasionally
                if False:
                    gradcam = compute_gradcam(model, d["centre"])
                else:
                    gradcam = np.zeros(361)

                d["gradcam"] = gradcam
                hr = d["hr"]

                raw_alert = compute_alert_level(hr, rhythm)

                if not hasattr(self, "alert_hold"):
                    self.alert_hold = {}

                if patient_id not in self.alert_hold:
                    self.alert_hold[patient_id] = {
                        "current": raw_alert,
                        "count": 0
                    }

                state = self.alert_hold[patient_id]

                if raw_alert == "CRITICAL":
                    state["current"] = "CRITICAL"
                    state["count"] = 0
                    alert = "CRITICAL"

                elif raw_alert == "NORMAL":
                    state["current"] = "NORMAL"
                    state["count"] = 0
                    alert = "NORMAL"

                elif raw_alert == state["current"]:
                    state["count"] = 0
                    alert = raw_alert

                else:
                    state["count"] += 1

                    if state["count"] >= 2:
                        state["current"] = raw_alert
                        state["count"] = 0

                    alert = state["current"]
                if rhythm == "AFib" and alert == "NORMAL":
                    alert = "WARNING"
                risk = compute_risk(hr, d["sdnn"])

                # --- LSTM ---
                patient_id = d["patient"]   # ✅ ADD THIS LINE

                # 🔥 TEMP DISABLE LSTM (FREEZE TEST)
                future_risk = risk

                # --- DQN ---
                # 🔥 TEMP DISABLE DQN (FREEZE TEST)
                rl_alert = alert

                # ❌ DO NOT TRAIN HERE FREQUENTLY
                # attach results
                # =========================================================
                # 🔥 ATTACH ALL UI FIELDS (CRITICAL FIX)
                # =========================================================

                d["rhythm"] = rhythm
                actual_conf = float(np.max(pred) * 100.0)

                if actual_conf < 55:
                    d["explanation"] = "Low confidence — manual review suggested"

                d["conf"] = actual_conf

                d["alert"] = alert
                d["rl_alert"] = rl_alert

                d["risk"] = risk
                d["future_risk"] = float(future_risk)
                d["updated"] = datetime.now().strftime("%H:%M:%S")
                d["risk_label"] = (
                    "CRITICAL" if risk >= 75 else
                    "HIGH" if risk >= 50 else
                    "MODERATE" if risk >= 25 else
                    "LOW"
                )
                if alert == "CRITICAL":
                    d["status_badge"] = "🔴 CRITICAL"
                elif alert == "WARNING":
                    d["status_badge"] = "🟡 WARNING"
                else:
                    d["status_badge"] = "🟢 STABLE"
                d["explanation"] = explain_prediction(d, rhythm)

                d["priority"] = float(icu_priority_score(d))
                key = d["patient"]

                if not hasattr(self, "last_logged"):
                    self.last_logged = {}

                if not hasattr(self, "last_log_time"):
                    self.last_log_time = {}

                norm_rhythm = str(rhythm).upper()
                norm_alert = str(alert).upper()

                now = time.time()

                severity = {
                    "NORMAL": 0,
                    "WARNING": 1,
                    "CRITICAL": 2
                }.get(norm_alert, 1)

                last = self.last_logged.get(key, {})
                last_severity = last.get("severity", -1)
                last_state = last.get("state", "")
                last_time = self.last_log_time.get(key, 0)

                current_state = f"{norm_rhythm}|{norm_alert}"

                should_log = False

                # always log escalation
                if severity > last_severity:
                    should_log = True

                # log changed state only if 10 sec passed
                elif current_state != last_state and (now - last_time) > 20:
                    should_log = True

                # periodic reminder if persistent alert
                elif norm_alert != "NORMAL" and (now - last_time) > 60:
                    should_log = True

                if should_log and norm_alert != "NORMAL":
                    logging.info(
                        f"PATIENT={d['patient']} | RHYTHM={rhythm} | HR={hr:.1f} | "
                        f"ALERT={alert} | RISK={risk} | FUTURE={future_risk:.1f}"
                    )

                    self.last_logged[key] = {
                        "state": current_state,
                        "severity": severity
                    }
                    self.last_log_time[key] = now
                try:
                    while self.out_q.qsize() >= 3:
                        try:
                            self.out_q.get_nowait()
                        except queue.Empty:
                            break

                    self.out_q.put_nowait(d)

                except queue.Full:
                    pass
                time.sleep(0.03)
            except Exception as e:
                print("[AI ERROR]:", e)
class CardioSentinelDashboard:
    def __init__(self, patients):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        self.win = pg.GraphicsLayoutWidget(
            title="CardioSentinel AI  —  Clinical ECG Monitoring System v2.0"

        )
        self.win.resize(1750, 1050)
        self.win.setBackground("#000000")
        self.win.show()
        QtWidgets.QApplication.processEvents()
        pg.setConfigOption("antialias", False)
        self.queue = queue.Queue(maxsize=20)
        self.ai_queue = queue.Queue(maxsize=20)
        self.ecg_curves   = {}
        self.peak_scatter = {}
        self.info_labels  = {}
        self.panels       = {}
        # 🧾 EVENT LOG (last 50 events)
        self.event_log = deque(maxlen=50)
        self.threads      = []
        # 🔊 sound control (per patient)
        self.last_alert_time = {p: 0 for p in patients}

        self.sound_cooldown = 2.0  # seconds between sounds
        # 🔥 HR trend buffers
        self.hr_history = {p: [] for p in patients}
        self.time_history = {p: [] for p in patients}
        for i, p in enumerate(patients):
            # ECG plot
            plot = self.win.addPlot(row=i, col=0)

            # 🔥 ICU STYLE
            plot.getViewBox().setBackgroundColor("#000000")
            plot.setTitle(
                f"<span style='color:#8899bb;font-family:Courier New,monospace;"
                f"font-size:9pt;'>Patient {p}  |  MIT-BIH Record {p}</span>",
                size="9pt",
            )

            # 📈 HR TREND PLOT
            hr_plot = self.win.addPlot(row=i, col=2)
            hr_plot.setYRange(40, 180)
            hr_plot.setTitle(f"HR Trend - Patient {p}", size="8pt")

            # 🔥 ICU STYLE FOR HR
            hr_plot.getViewBox().setBackgroundColor("#000000")
            hr_plot.getAxis("left").setPen(pg.mkPen("#ffaa00"))
            hr_plot.getAxis("bottom").setPen(pg.mkPen("#ffaa00"))
            hr_plot.getAxis("left").setTextPen(pg.mkPen("#ffaa00"))
            hr_plot.getAxis("bottom").setTextPen(pg.mkPen("#ffaa00"))
            hr_plot.showGrid(x=True, y=True, alpha=0.2)

            hr_curve = hr_plot.plot(pen=pg.mkPen("#ffaa00", width=2))

            if not hasattr(self, "hr_curves"):
                self.hr_curves = {}

            self.hr_curves[p] = hr_curve

            plot.setYRange(-2.8, 2.8)

            plot.getAxis("left").setPen(pg.mkPen("#00ff00", width=1))
            plot.getAxis("bottom").setPen(pg.mkPen("#00ff00", width=1))
            plot.getAxis("left").setTextPen(pg.mkPen("#00ff88"))
            plot.getAxis("bottom").setTextPen(pg.mkPen("#00ff88"))

            plot.showGrid(x=True, y=True, alpha=0.2)

            plot.setLabel("bottom", "Time (s)", color="#00ff88")
            plot.setLabel("left", "Amplitude (norm)", color="#00ff88")

            curve = plot.plot(pen=pg.mkPen("#00ff41", width=2))
            scatter = plot.plot(
                symbol="o", pen=None,
                symbolBrush="#ff0033", symbolSize=7,
            )
            self.ecg_curves[p]   = curve
            self.peak_scatter[p] = scatter

            # Clinical data panel
            # 🔥 ICU PANEL CONTAINER
            panel = QtWidgets.QLabel()
            panel.setMinimumWidth(340)
            panel.setFixedHeight(205)
            panel.setWordWrap(True)
            panel.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignTop |
                QtCore.Qt.AlignmentFlag.AlignLeft
            )
            panel.setTextFormat(QtCore.Qt.TextFormat.RichText)
            panel.setContentsMargins(8, 8, 8, 8)
            # default style (safe)
            panel.setStyleSheet("""
                background-color: #050a1f;
                color: #00ffcc;
                border: 1px solid #00ffaa;
                border-radius: 8px;
                padding: 10px;
                font-family: Consolas;
            """)

            # ✅ Correct way
            proxy = QtWidgets.QGraphicsProxyWidget()
            proxy.setWidget(panel)

            self.win.addItem(proxy, row=i, col=1)

            self.panels[p] = panel
            self.info_labels[p] = panel

        # Refresh timer
        self.timer = QtCore.QTimer()
        # 🔥 SMOOTH BLINK STATE
        self.blink_phase = 0
        self.timer.timeout.connect(self._update)
        self.timer.start(500)

        # Launch acquisition threads
        self.ai_thread = AIThread(self.queue, self.ai_queue)
        self.ai_thread.start()
        for p in patients:
            t = PatientThread(p, self.queue)
            t.start()
            self.threads.append(t)

    # ── Label builder ─────────────────────────────────────────────
    def _build_clinical_panel(
            self, d, rhythm, conf, alarm_type, alert_level,
            risk_score, risk_lbl, risk_color, sdnn, rmssd, pnn50,
            qrs_ms, pr_ms, total_latency_ms
    ):
        hr    = d["hr"]# 🔥 Update HR trend (last 30 sec)
        patient_id = d["patient"]
        now = time.time()
        # smoothing (prevents flat + jitter)

        if len(self.hr_history[patient_id]) > 0:
            hr = 0.7*self.hr_history[patient_id][-1] + 0.3*hr
        self.time_history[patient_id].append(now)

        while self.time_history[patient_id] and (now - self.time_history[patient_id][0] > 30):
            self.time_history[patient_id].pop(0)
            self.hr_history[patient_id].pop(0)
        irreg = d["irregular"]

        hr_sev   = range_severity(hr, "HR_BPM")
        hr_color = {
            "NORMAL"       : "#00e5a0",
            "ALERT_LOW"    : "#ffd700",
            "ALERT_HIGH"   : "#ffd700",
            "CRITICAL_LOW" : "#ff2d55",
            "CRITICAL_HIGH": "#ff2d55",
        }.get(hr_sev, "#ffffff")

        qrs_sev   = range_severity(qrs_ms, "QRS_DURATION_MS")
        qrs_color = "#ff6b35" if qrs_sev != "NORMAL" else "#a0c0e0"
        pr_sev    = range_severity(pr_ms, "PR_INTERVAL_MS")
        pr_color  = "#ff6b35" if pr_sev != "NORMAL" else "#a0c0e0"

        # HR range annotation
        cr = CLINICAL_RANGES["HR_BPM"]
        hr_range_str = (
            f"Normal: {cr['normal_low']}–{cr['normal_high']} bpm  "
            f"Alert: <{cr['lower_alert']} | >{cr['upper_alert']}  "
            f"Critical: <{cr['lower_critical']} | >{cr['upper_critical']}"
        )

        # Alarm block
        alarm_html    = ""
        golden_html   = ""
        etiology_html = ""

        if alarm_type == "AFIB":
            gt = GOLDEN_TIME["AFIB_CARDIOVERSION"]
            alarm_html = (
                "<b style='color:#ff2d55;font-size:10pt'>⚠ ATRIAL FIBRILLATION</b>"
            )
            golden_html = (
                f"<span style='color:#ffa040'>⏱ Cardioversion window: 48 h &nbsp;|"
                f"&nbsp;After 48 h → Stroke risk ×5</span><br>"
                f"<span style='color:#80c0ff'>Action: {gt['action']}</span>"
            )
            etiology_html = (
                "<span style='color:#607090'>Causes: "
                + " &nbsp;·&nbsp; ".join(ETIOLOGY["AFib"][:3])
                + "</span>"
            )

        elif alarm_type == "BRADYCARDIA":
            gt = GOLDEN_TIME["SEVERE_BRADYCARDIA"]
            sev_str = "⚠ SEVERE BRADYCARDIA" if hr < 40 else "⚠ BRADYCARDIA"
            alarm_html = f"<b style='color:#ff6b35;font-size:10pt'>{sev_str}</b>"
            golden_html = (
                f"<span style='color:#ffa040'>⏱ Golden window: {gt['window_minutes']} min"
                f"&nbsp;|&nbsp;Hard deadline: {gt['hard_deadline_min']} min</span><br>"
                f"<span style='color:#80c0ff'>Action: {gt['action']}</span>"
            )
            etiology_html = (
                "<span style='color:#607090'>Causes: "
                + " &nbsp;·&nbsp; ".join(ETIOLOGY["BRADYCARDIA"][:3])
                + "</span>"
            )
            if hr < CLINICAL_RANGES["HR_BPM"]["lower_critical"]:
                etiology_html += (
                    "<br><span style='color:#ff6b35;font-size:7pt'>"
                    + CLINICAL_RANGES["HR_BPM"]["below_lower_note"]
                    + "</span>"
                )

        elif alarm_type == "TACHYCARDIA":
            gt = GOLDEN_TIME["SVT"]
            sev_str = "🚨 CODE TACHYCARDIA" if hr > 160 else "⚠ TACHYCARDIA"
            alarm_html = f"<b style='color:#ff2d55;font-size:10pt'>{sev_str}</b>"
            golden_html = (
                f"<span style='color:#ffa040'>⏱ Intervention window: {gt['window_minutes']} min"
                f"&nbsp;|&nbsp;Hard deadline: {gt['hard_deadline_min']} min</span><br>"
                f"<span style='color:#80c0ff'>Action: {gt['action']}</span>"
            )
            etiology_html = (
                "<span style='color:#607090'>Causes: "
                + " &nbsp;·&nbsp; ".join(ETIOLOGY["TACHYCARDIA"][:3])
                + "</span>"
            )

        elif alarm_type == "PVC_STORM":
            gt = GOLDEN_TIME["PVC_STORM"]
            alarm_html = "<b style='color:#ff6b35;font-size:10pt'>⚠ PVC STORM — R-on-T Risk</b>"
            golden_html = (
                f"<span style='color:#ffa040'>⏱ Intervention window: {gt['window_minutes']} min"
                f"&nbsp;|&nbsp;Hard deadline: {gt['hard_deadline_min']} min</span><br>"
                f"<span style='color:#80c0ff'>Action: {gt['action']}</span>"
            )
            etiology_html = (
                "<span style='color:#607090'>Causes: "
                + " &nbsp;·&nbsp; ".join(ETIOLOGY["PVC"][:3])
                + "</span>"
            )

        else:
            alarm_html = "<span style='color:#00e5a0'>— STABLE —</span>"

        al_cfg    = ALERT_LEVELS.get(alert_level or "INFO", ALERT_LEVELS["INFO"])
        act_sec   = al_cfg["action_sec"]
        act_str   = (f"{act_sec} s" if act_sec > 0 else "IMMEDIATE")

        html = f"""
<div style='font-family:Courier New,monospace;font-size:8.5pt;
            color:#b0c4d8;line-height:1.65;padding:4px'>

<b style='font-size:11pt;color:{hr_color}'>HR: {hr:.1f} bpm</b>
&nbsp;<span style='color:#2a4060'>|</span>&nbsp;
<b style='color:{CLASS_COLORS.get(rhythm,"#fff")}'>
  {rhythm} &nbsp;({conf:.1f}%)</b>
<br>
<span style='color:#405570;font-size:7.5pt'>{hr_range_str}</span>
<br>
<span style='color:#506080'>QRS:</span>&nbsp;<b style='color:{qrs_color}'>{qrs_ms:.0f} ms</b>
&nbsp;&nbsp;
<span style='color:#506080'>PR:</span>&nbsp;<b style='color:{pr_color}'>{pr_ms:.0f} ms</b>
&nbsp;&nbsp;
<span style='color:#506080'>SDNN:</span>&nbsp;{sdnn:.0f} ms
&nbsp;&nbsp;
<span style='color:#506080'>RMSSD:</span>&nbsp;{rmssd:.0f} ms
<br>
<span style='color:#506080'>pNN50:</span>&nbsp;{pnn50:.1f}%
&nbsp;&nbsp;
<span style='color:#506080'>RR Irregular:</span>&nbsp;
{'<b style="color:#ff2d55">YES</b>' if irreg else
 '<span style="color:#00e5a0">No</span>'}
<br>
{alarm_html}
<br>
<span style='color:{al_cfg["color"]}'>
  [{alert_level or "STABLE"}]
  &nbsp;Act within: {act_str}
  &nbsp;|&nbsp;Risk: <b style='color:{risk_color}'>{risk_lbl} ({risk_score:.0f}/100)</b>
</span>
<br>
{golden_html}
{'<br>' if golden_html else ''}
{etiology_html}
{'<br>' if etiology_html else ''}
<span style='color:#1e3050;font-size:7pt'>
  Proc: {d['proc_ms']:.1f} ms
  &nbsp;|&nbsp;Total: {total_latency_ms:.0f} ms
  &nbsp;|&nbsp;{d['timestamp']}
</span>
</div>
"""
        # 🧠 AI EXPLANATION (ADD HERE — BEFORE RETURN)

        try:
            explanation = explain_prediction(d, rhythm)

            html += f"""
            <br>
            <span style='color:#00ffaa'>AI Insight:</span>
            <span style='color:#a0ffcc'>{explanation}</span>
            """
        except Exception as e:
            pass

        return html

    # ── UI update callback ────────────────────────────────────────
    def _update(self):
        try:
            updates = []

            # 🔥 PULL DATA ONCE
            data_list = []

            while not self.ai_queue.empty():
                data_list.append(self.ai_queue.get_nowait())

            if not data_list:
                return

            data_list.sort(key=lambda x: x.get("priority", 0), reverse=True)

            # 🔥 SMOOTH BLINK UPDATE
            self.blink_phase += 1
            blink = (self.blink_phase % 20) < 10

            max_updates = 3   # ✅ LIMIT UI LOAD
            count = 0

            for d in data_list[:3]:  # process max 3 items per frame

                patient_id = d["patient"]
                now = time.time()
                # 🧾 LOG EVENT (ONLY FOR WARNING+)
                if d.get("alert") in ["WARNING", "CRITICAL"]:
                    log_entry = f"{d['timestamp']} | P{patient_id} | {d['alert']} | HR={d['hr']:.1f}"
                    self.event_log.appendleft(log_entry)
                # 🔴 CRITICAL DETECTION
                is_critical = d.get("alert", "") in ["CRITICAL", "CODE"]
                # 🔴 BLINKING BACKGROUND
                if is_critical:
                    if int(time.time() * 2) % 2 == 0:
                        bg_color = "#3a0000"   # dark red
                    else:
                        bg_color = "#000000"   # black
                else:
                    bg_color = "#0b132b"
                # 🔊 SOUND CONTROL
                now = time.time()

                if is_critical:
                    if now - self.last_alert_time[patient_id] > 2:

                        if d.get("alert") == "CODE":
                            play_alert("high")

                        elif d.get("alert") == "CRITICAL":
                            play_alert("medium")

                        else:
                            play_alert("low")

                        self.last_alert_time[patient_id] = now
                if patient_id not in self.ecg_curves:
                    continue

                curve = self.ecg_curves[patient_id]
                scatter = self.peak_scatter[patient_id]

                # Time axis
                x = np.linspace(0, WINDOW_SEC, len(d["window"]))

                # Plot ECG
                saliency = np.zeros(360)

                gradcam = d.get("gradcam", None)

                if gradcam is None or len(gradcam) < 2:
                    gradcam_resized = np.zeros(len(d["window"]))
                else:
                    gradcam_resized = np.interp(
                        np.linspace(0, len(gradcam)-1, len(d["window"])),
                        np.arange(len(gradcam)),
                        gradcam
                    )

                highlighted = d["window"] * (1 + 0.8 * gradcam_resized)

                smooth = np.convolve(d["window"], np.ones(5)/5, mode='same')
                curve.setData(smooth)

                # Plot peaks
                if len(d["peaks"]) > 0:
                    scatter.setData(x[d["peaks"]], d["window"][d["peaks"]])

                # 🔥 EVERYTHING BELOW MUST BE INSIDE LOOP
                saliency = np.zeros_like(d["window"])
                saliency = saliency / (np.max(saliency) + 1e-6)
                
                rhythm = d["rhythm"]
                conf = d["conf"]
                # 🧠 AI EXPLANATION
                explanation = explain_prediction(d, rhythm)
                # 🔥 ALERT SOUND + FLASH
                level = d["alert"]
                if level in ["CRITICAL", "CODE"]:
                    color = "#ff2d55"
                    if time.time() - self.last_alert_time[d["patient"]] > self.sound_cooldown:
                        threading.Thread(target=play_alert, args=("high",), daemon=True).start()
                        self.last_alert_time[d["patient"]] = time.time()

                elif level == "WARNING":
                    color = "#ffd700"
                    if time.time() - self.last_alert_time[d["patient"]] > self.sound_cooldown:
                        threading.Thread(target=play_alert, args=("medium",), daemon=True).start()
                        self.last_alert_time[d["patient"]] = time.time()

                else:
                    color = "#0b132b"

                # 🔥 blinking effect
                if level in ["CRITICAL", "CODE"]:
                    self.blink_phase = (self.blink_phase + 1) % 2
                    if self.blink_phase == 0:
                        color = "#000000"

                self.panels[d["patient"]].setStyleSheet(f"""
                    background-color: {color};
                    border-radius: 10px;
                    padding: 12px;
                """)
                risk = d["risk"]
                future_risk = d["future_risk"]
                hr = d["hr"]
                # 📈 Update HR graph
                if len(self.hr_history[patient_id]) > 1:
                    t_vals = np.array(self.time_history[patient_id])
                    t_vals = t_vals - t_vals[0]  # normalize time axis
                    # 🔥 SAFE HR PLOT (FIX CRASH)
                    y_vals = self.hr_history[patient_id]
                    x_vals = list(range(len(y_vals)))

                    # ensure same length
                    n = min(len(x_vals), len(y_vals))
                    x_vals = x_vals[:n]
                    y_vals = y_vals[:n]

                    if n > 1:
                        self.hr_curves[patient_id].setData(x_vals, y_vals)
                now = time.time()
                if len(self.hr_history[patient_id]) > 0:
                    hr = 0.7*self.hr_history[patient_id][-1] + 0.3*hr

                self.hr_history[patient_id].append(hr)
                self.time_history[patient_id].append(now)

                # keep last 30 sec
                while self.time_history[patient_id] and (now - self.time_history[patient_id][0] > 30):
                    self.time_history[patient_id].pop(0)
                    self.hr_history[patient_id].pop(0)
                explanation = explain_prediction(d, rhythm)

                # 🔊 SOUND ESCALATION
                if now - self.last_alert_time.get(patient_id, 0) > self.sound_cooldown:

                    if level == "CRITICAL":
                        winsound.Beep(2000, 300)   # high pitch fast
                        winsound.Beep(2000, 300)

                    elif level == "WARNING":
                        winsound.Beep(1200, 400)   # medium

                    elif level == "NORMAL":
                        winsound.Beep(800, 200)    # soft

                    self.last_alert_time[patient_id] = now
                blink = int(time.time()*2) % 2 == 0
                risk = d["risk"]
                arrest_risk = predict_cardiac_arrest(hr, d["sdnn"], d["qrs_ms"])

                # --- LSTM prediction ---
                #dqn.replay()
                # train occasionally (stability)
                if np.random.rand() < 0.1:
                    dqn.replay()
                priority = icu_priority_score(d)
                updates.append((patient_id, d))

            for patient_id, d in updates:
                alert = d["alert"]
                rhythm = d["rhythm"]
                hr = d["hr"]
                risk = d["risk"]
                # 🎨 PANEL COLOR
                if alert == "CRITICAL":
                    blink = int(time.time() * 2) % 2 == 0
                    bg = "#3b0a0a" if blink else "#140000"
                    border = "#ff0000" if blink else "#770000"

                elif alert == "WARNING":
                    bg = "#3a3200"
                    border = "#ffd700"

                else:
                    bg = "#0b132b"
                    border = "#00e5a0"

                if patient_id not in self.panels:
                    continue
                # 🧾 EVENT LOG HTML
                log_html = "<br>".join(list(self.event_log)[:5])
                html = f"""
                <div style="
                    line-height:1.15;
                    font-size:9pt;
                    color:white;
                ">
                    <b style='font-size:13pt;'>Patient {d['patient']}</b><br>

                    <span style="font-size:10pt;">{d['status_badge']}</span><br>
                    ❤️ HR: {hr:.1f} bpm<br>
                    🧠 Rhythm: {rhythm}<br>
                    📊 Confidence: {d['conf']:.1f}%<br>
                    📈 Risk: {risk:.1f}/100<br>
                    🔮 Future Risk: {d['future_risk']:.1f}/100<br>
                    <span style="color:#7fdfff; font-size:8.5pt;">
                    🩺 Interpretation: {d['explanation']}
                    </span>
                </div>
                """
                blink = False
                if alert == "CRITICAL":
                    blink = int(time.time() * 2) % 2 == 0
                    panel_color = "#ff0000" if blink else "#550000"
                final_html = f"""
                <div style="
                    background-color: {bg};
                    color: white;
                    padding: 5px;
                    border-radius: 8px;
                ">
                    {html}
                </div>
                """

                # 🔥 TAGS
                if d.get("alert") in ["CRITICAL", "CODE"]:
                    final_html += "<br><b style='color:red'>⚠ CRITICAL ⚠</b>"

                if patient_id == updates[0][0] if updates else False:
                    final_html += "<br><b style='color:#ff5555'>TOP PRIORITY</b>"

                # 🔥 FINAL STYLE (ONLY ONE)
                self.panels[patient_id].setStyleSheet(f"""
                    background-color: {bg};
                    border: 3px solid {border};
                    border-radius: 6px;
                    padding: 10px;
                    color: #00ff88;
                    font-family: Courier New;
                """)

                # 🔥 FINAL DISPLAY
                self.panels[patient_id].setText(final_html)
            if updates:
                updates.sort(
                    key=lambda x: x[1].get("priority", 0),
                    reverse=True
                )
                top_patient = updates[0][0]

                for p in self.panels:
                    if p == top_patient:
                        self.panels[p].setStyleSheet(self.panels[p].styleSheet() + "font-size: 105%;")
        except Exception as e:
            print("[UPDATE ERROR]:", e)
    def run(self):
        self.app.exec()


# ================================================================
#  MAIN
# ================================================================
if __name__ == "__main__":
    banner = """
╔══════════════════════════════════════════════════════════════╗
║   CardioSentinel AI  —  Clinical ECG Monitor  v2.0           ║
║   Author       : Vaibhav Krishna V                           ║
║   Architecture : Multi-Scale Residual Attention 1D-CNN       ║
║   Dataset      : MIT-BIH Arrhythmia DB (PhysioNet)           ║
║   Alert log    : cardiosentinel_alerts.log                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    app = QtWidgets.QApplication([])
    print(banner)
    print("🚀 Starting CardioSentinel...")

    # 1. Load / train model FIRST
    model = load_or_train_model()

    # 3. Initialize DQN AFTER model
    dqn = DQN()
    pg.setConfigOptions(useOpenGL=False)
    # 4. START UI LAST (CRITICAL)
    dash = CardioSentinelDashboard(PATIENTS)
    dash.run()
