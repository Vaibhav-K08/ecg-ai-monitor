# 🫀 Clinical Grade Real-Time ECG AI Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![PyQtGraph](https://img.shields.io/badge/PyQtGraph-Live_Dashboard-41CD52?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A multi-patient, real-time cardiac monitoring system with 4-class AI rhythm classification, clinical AFib detection, and a live color-coded waveform dashboard — all on CPU.*

</div>

---

## 📌 Project Overview

This system streams ECG signals from the MIT-BIH Arrhythmia Database and performs continuous cardiac analysis across **multiple patients simultaneously** using a multithreaded architecture. It combines a trained 1D CNN classifier with deterministic clinical logic to detect dangerous rhythms in real time, visualized on a live PyQtGraph dashboard.

No GPU required. Fully CPU-deployable.

---

## 🏗️ System Architecture

```
MIT-BIH ECG Stream (per patient thread)
        ↓
  Bandpass Filter (0.5–40 Hz, Butterworth)
        ↓
  R-Peak Detection (energy envelope + find_peaks)
        ↓
  ┌──────────────────────────────────────────┐
  │  Parallel Analysis                        │
  │  ├── Heart Rate (median RR interval)      │
  │  ├── HRV (SDNN + RMSSD)                  │
  │  └── AFib Detection (HRV thresholds)      │
  └──────────────────────────────────────────┘
        ↓
  1D CNN Classifier → 4-class rhythm prediction
        ↓
  Hybrid Clinical Validator
  ├── AFib:        SDNN > 0.12s AND RMSSD > 0.1s
  ├── Tachycardia: HR > 120 BPM + non-Normal rhythm
  └── Bradycardia: HR < 45 BPM
        ↓
  Live PyQtGraph Dashboard
  ├── Color-coded waveform (🟢 Stable / 🔴 Alarm)
  ├── Per-patient HR, Rhythm, Confidence, Status
  └── Real-time peak markers
```

---

## 🧠 AI Model — 4-Class 1D CNN

The classifier is trained on real annotated beats from MIT-BIH using actual beat-level labels:

| Class | MIT-BIH Symbol | Description |
|---|---|---|
| Normal | `N` | Normal sinus rhythm |
| PVC | `V` | Premature Ventricular Contraction |
| AFib | `A` | Atrial Fibrillation / flutter |
| Other | everything else | Unclassified arrhythmia |

**Training records:** 100, 101, 102, 103, 104, 105, 106  
**Input shape:** 360 samples (1 second centered on beat)  
**Architecture:** Conv1D(32) → BN → MaxPool → Conv1D(64) → GAP → Dense(64) → Softmax(4)  
**Epochs:** 6 | **Batch size:** 128 | **Optimizer:** Adam

> Model is saved after first training run as `ecg_clinical_cpu.keras` and reloaded automatically on subsequent runs — no retraining needed.

---

## ⚡ Clinical Logic Layer

On top of AI predictions, the system applies **hard clinical rules** that override the model when physiological thresholds are breached:

```python
# AFib: irregular RR intervals detected via HRV
if SDNN > 0.12s and RMSSD > 0.10s → AFIB ALARM

# Tachycardia: fast heart rate + non-normal rhythm
if HR > 120 BPM and rhythm != "Normal" → TACHYCARDIA ALARM

# Bradycardia: dangerously slow heart rate
if 0 < HR < 45 BPM → BRADYCARDIA ALARM
```

This hybrid approach prevents isolated model misclassifications from triggering false alarms.

---

## 📊 Dashboard

The live PyQtGraph dashboard displays per patient:
- **ECG waveform** — rendered at 150ms refresh rate
- **R-peak markers** — red dots overlaid on waveform
- **Waveform color** — green (stable) / red (alarm)
- **Sidebar** — HR (BPM), Rhythm class, Confidence (%), Status

Each patient runs in its own thread, streaming and updating independently.

---

## ✅ Key Features

- 🔴 **Multi-patient simultaneous monitoring** via threading
- 🧠 **4-class CNN** trained on real annotated MIT-BIH beats
- 📊 **Live color-coded PyQtGraph dashboard** — red on clinical alarm
- ❤️ **HRV-based AFib detection** — SDNN + RMSSD thresholds
- ⚡ **Model caching** — trains once, reloads on all future runs
- 🏥 **CPU-only deployment** — no GPU dependency

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | 1D CNN training and inference |
| `WFDB` | MIT-BIH ECG streaming and annotation loading |
| `SciPy` | Butterworth bandpass filter, peak detection |
| `NumPy` | Signal processing and HRV computation |
| `PyQtGraph` | Real-time waveform rendering |
| `PyQt5` | GUI application framework |
| `threading` | Parallel per-patient data streams |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecg-ai-monitor.git
cd ecg-ai-monitor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python ecg_monitor.py
```

On first run, the model trains automatically on MIT-BIH records (downloaded via WFDB) and saves to `ecg_clinical_cpu.keras`. Subsequent runs load the saved model instantly.

---

## 📁 Project Structure

```
ecg-ai-monitor/
├── ecg_monitor.py          # Complete pipeline — run this
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏥 Applications

**Industrial:** Portable ECG monitoring devices · Edge AI healthcare platforms · Real-time biomedical signal analytics · CPU-based medical inference engines · Clinical monitoring interfaces

**Societal:** Early detection of cardiac abnormalities · Affordable monitoring in low-resource regions · Remote healthcare and telemedicine · Continuous monitoring for elderly populations

---

## 👤 Author

**Vaibhav Krishna V**  
Electronics & Communication Engineer  
📧 vaibhavkv078@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
