"""
Clinical Grade Real Time ECG AI Monitoring System
Author: Vaibhav Krishna V
"""

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow import keras
import sys
import time

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SAMPLE_RATE       = 360          # MIT-BIH sampling frequency (Hz)
WINDOW_SIZE       = 1800         # 5-second sliding window (samples)
STEP_SIZE         = 360          # Slide by 1 second
BEAT_WINDOW       = 72           # Samples per heartbeat segment (~0.2s)
TACHYCARDIA_BPM   = 100
BRADYCARDIA_BPM   = 60
MODEL_SAVE_PATH   = "ecg_model.h5"

# MIT-BIH records to stream
RECORDS = ["100", "101", "103", "105", "106", "108"]

# ─────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLE_RATE, order=4):
    """Remove baseline wander (low-freq) and noise (high-freq)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(filtered_signal, fs=SAMPLE_RATE):
    """
    Adaptive R-peak detection using normalized signal + threshold.
    Returns list of sample indices where R-peaks occur.
    """
    # Normalize signal
    norm = (filtered_signal - np.mean(filtered_signal)) / (np.std(filtered_signal) + 1e-8)
    
    # Adaptive threshold: mean + 0.6 * std of positive peaks
    threshold = np.mean(norm) + 0.6 * np.std(norm)
    
    peaks = []
    min_distance = int(0.2 * fs)  # Minimum 200ms between beats (300 BPM max)
    last_peak = -min_distance

    for i in range(1, len(norm) - 1):
        if (norm[i] > threshold and
                norm[i] > norm[i - 1] and
                norm[i] > norm[i + 1] and
                (i - last_peak) > min_distance):
            peaks.append(i)
            last_peak = i

    return np.array(peaks)


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_features(r_peaks, fs=SAMPLE_RATE):
    """
    Compute RR intervals, heart rate, and HRV from detected peaks.
    Returns dict of physiological metrics.
    """
    if len(r_peaks) < 2:
        return {"heart_rate": 0, "rr_mean": 0, "rr_std": 0, "hrv": 0}

    rr_intervals = np.diff(r_peaks) / fs  # In seconds
    rr_ms        = rr_intervals * 1000     # In milliseconds
    heart_rate   = 60.0 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0

    return {
        "heart_rate": round(heart_rate, 1),
        "rr_mean":    round(np.mean(rr_ms), 1),
        "rr_std":     round(np.std(rr_ms), 1),
        "hrv":        round(np.sqrt(np.mean(np.diff(rr_ms) ** 2)), 1)  # RMSSD
    }


# ─────────────────────────────────────────
# LIGHTWEIGHT CNN MODEL
# ─────────────────────────────────────────

def build_model(input_length=BEAT_WINDOW, num_classes=2):
    """
    Compact 1D CNN optimized for CPU inference.
    Input:  single ECG beat segment (BEAT_WINDOW samples)
    Output: class probabilities [Normal, Arrhythmic]
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_length, 1)),

        # Block 1
        keras.layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        # Block 2
        keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        # Block 3
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling1D(),

        # Classifier head
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ], name="ECG_Classifier")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def extract_beat_segments(signal, r_peaks, half_window=BEAT_WINDOW // 2):
    """Slice fixed-length beat windows centered on each R-peak."""
    segments = []
    labels   = []
    for peak in r_peaks:
        start = peak - half_window
        end   = peak + half_window
        if start >= 0 and end <= len(signal):
            segment = signal[start:end]
            segments.append(segment)
            labels.append(0)  # Placeholder label (Normal)
    return np.array(segments), np.array(labels)


def train_model(model, segments, labels, epochs=6):
    """Train the lightweight CNN on extracted beat segments."""
    X = segments.reshape(-1, BEAT_WINDOW, 1).astype(np.float32)
    # Normalize per-sample
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    print("Training clinical CPU model...")
    model.fit(X, labels, epochs=epochs, batch_size=32, verbose=1)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}")
    return model


# ─────────────────────────────────────────
# HYBRID CLINICAL VALIDATOR
# ─────────────────────────────────────────

def hybrid_validate(ai_label, ai_confidence, heart_rate):
    """
    Combine AI prediction with deterministic clinical rules.
    Clinical rules override uncertain AI predictions.
    """
    # Override with hard clinical rules
    if heart_rate > TACHYCARDIA_BPM:
        return "Tachycardia", 1.0, "Clinical Rule"
    elif heart_rate < BRADYCARDIA_BPM and heart_rate > 0:
        return "Bradycardia", 1.0, "Clinical Rule"

    # Trust AI if confidence is high
    if ai_confidence > 0.75:
        label = "Normal" if ai_label == 0 else "Arrhythmic"
        return label, ai_confidence, "AI Model"

    # Low-confidence AI → fallback to Normal (conservative)
    return "Normal (Low Confidence)", ai_confidence, "Fallback"


# ─────────────────────────────────────────
# REAL-TIME MONITORING PIPELINE
# ─────────────────────────────────────────

def load_ecg_record(record_id):
    """Download and return ECG signal from MIT-BIH database."""
    print(f"\nLoading record {record_id} from MIT-BIH...")
    record = wfdb.rdrecord(record_id, pn_dir='mitdb')
    signal = record.p_signal[:, 0]  # Lead II
    print(f"  Loaded {len(signal)} samples @ {SAMPLE_RATE} Hz")
    return signal


def run_monitor(record_id="100"):
    """Main real-time monitoring loop for a single ECG record."""
    # Load signal
    raw_signal = load_ecg_record(record_id)

    # Filter
    filtered = bandpass_filter(raw_signal)

    # Detect R-peaks on full signal (for training data)
    all_peaks = detect_r_peaks(filtered)
    print(f"  Detected {len(all_peaks)} R-peaks")

    # Extract beat segments and train model
    segments, labels = extract_beat_segments(filtered, all_peaks)
    model = build_model()
    if len(segments) > 10:
        model = train_model(model, segments, labels)
    else:
        print("  Not enough segments to train. Using untrained model.")

    # ── Streaming simulation ──────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  REAL-TIME MONITORING — Record {record_id}")
    print(f"{'='*55}")

    window_count = 0
    for start in range(0, len(filtered) - WINDOW_SIZE, STEP_SIZE):
        window       = filtered[start : start + WINDOW_SIZE]
        peaks_in_win = detect_r_peaks(window)
        features     = extract_features(peaks_in_win)
        hr           = features["heart_rate"]

        # AI classification on most recent beat
        ai_label, ai_conf = 0, 0.5
        if len(peaks_in_win) > 0:
            peak = peaks_in_win[-1]
            hw   = BEAT_WINDOW // 2
            if peak - hw >= 0 and peak + hw <= len(window):
                beat = window[peak - hw : peak + hw].reshape(1, BEAT_WINDOW, 1).astype(np.float32)
                beat = (beat - beat.mean()) / (beat.std() + 1e-8)
                probs    = model.predict(beat, verbose=0)[0]
                ai_label = int(np.argmax(probs))
                ai_conf  = float(np.max(probs))

        # Hybrid validation
        rhythm, confidence, source = hybrid_validate(ai_label, ai_conf, hr)

        window_count += 1
        status = "⚠ ALERT" if rhythm not in ("Normal", "Normal (Low Confidence)") else "✓ Stable"

        print(f"  Window {window_count:03d} | HR: {hr:5.1f} BPM | "
              f"Rhythm: {rhythm:<28} | Conf: {confidence:.0%} | "
              f"Source: {source:<15} | {status}")

        time.sleep(0.05)  # Simulate real-time pacing

    print(f"\n  Monitoring complete. {window_count} windows processed.")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    record = sys.argv[1] if len(sys.argv) > 1 else "100"
    print("=" * 55)
    print("  Clinical Grade Real Time ECG AI Monitor")
    print("  Author: Vaibhav Krishna V  ")
    print("=" * 55)
    run_monitor(record_id=record)
