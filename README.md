# 🫀 Clinical Grade Real Time ECG AI Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![PyQtGraph](https://img.shields.io/badge/PyQtGraph-Live_Dashboard-41CD52?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Monitors multiple patients simultaneously, classifies cardiac rhythms using a trained CNN, and flags clinical alarms in real time; runs entirely on CPU.*

</div>

---

## What This Does

This system streams ECG signals from the MIT-BIH Arrhythmia Database and runs continuous cardiac analysis across multiple patients at the same time using a multithreaded architecture. A trained 1D CNN handles rhythm classification, while a separate clinical logic layer catches dangerous conditions the model might miss. Everything shows up on a live PyQtGraph dashboard with color coded waveforms.

No GPU needed.

---

## How It Works

```
MIT-BIH ECG Stream (one thread per patient)
        ↓
  Bandpass Filter — 0.5 to 40 Hz (Butterworth)
        ↓
  R-Peak Detection — energy envelope + scipy find_peaks
        ↓
  Heart Rate  ←  median RR interval
  HRV         ←  SDNN and RMSSD
  AFib Check  ←  HRV threshold logic
        ↓
  CNN Classifier — 4 class rhythm prediction
        ↓
  Clinical Validator
  ├── AFib        : SDNN > 0.12s and RMSSD > 0.1s
  ├── Tachycardia : HR > 120 BPM with non-Normal rhythm
  └── Bradycardia : HR < 45 BPM
        ↓
  Live Dashboard
  ├── Waveform turns red on alarm, green when stable
  ├── HR, Rhythm, Confidence, Status per patient
  └── R-peak markers overlaid on waveform
```

---

## The AI Model

A 1D CNN trained on real annotated beats from MIT-BIH. Each beat is labeled using the actual annotation symbols from the database.

| Class | MIT-BIH Symbol | Meaning |
|---|---|---|
| Normal | `N` | Normal sinus rhythm |
| PVC | `V` | Premature Ventricular Contraction |
| AFib | `A` | Atrial Fibrillation |
| Other | rest | Unclassified arrhythmia |

**Training records:** 100, 101, 102, 103, 104, 105, 106  
**Input:** 360 samples centered on each beat (1 second)  
**Architecture:** Conv1D(32) → BatchNorm → MaxPool → Conv1D(64) → GlobalAvgPool → Dense(64) → Softmax(4)  
**Epochs:** 6 | **Batch size:** 128 | **Optimizer:** Adam

The model saves to `ecg_clinical_cpu.keras` after the first training run. Every run after that loads the saved model; no retraining.

---

## Clinical Logic

The model alone is not enough. On top of the CNN output, hard clinical rules run independently:

```python
# Irregular RR intervals → AFib
if SDNN > 0.12 and RMSSD > 0.10:
    alarm = "AFIB"

# Fast heart rate with abnormal rhythm → Tachycardia
if HR > 120 and rhythm != "Normal":
    alarm = "TACHYCARDIA"

# Dangerously slow heart rate → Bradycardia
if 0 < HR < 45:
    alarm = "BRADYCARDIA"
```

This prevents the model from silently missing a dangerous condition when it makes a low confidence prediction.

---

## Features

- Multiple patients monitored at the same time via threading
- CNN trained on real MIT-BIH beat annotations, not synthetic data
- AFib detection using HRV (SDNN + RMSSD) — not just the model output
- Waveform goes red the moment a clinical alarm triggers
- Model trains once and reloads on all future runs automatically
- Runs entirely on CPU, no GPU dependency

---

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | CNN training and inference |
| WFDB | MIT-BIH data streaming and annotation loading |
| SciPy | Butterworth filter and peak detection |
| NumPy | Signal processing and HRV computation |
| PyQtGraph | Live waveform rendering |
| PyQt5 | GUI framework |
| threading | Parallel patient data streams |

---

## Running It

```bash
git clone https://github.com/YOUR_USERNAME/ecg-ai-monitor.git
cd ecg-ai-monitor
pip install -r requirements.txt
python ecg_monitor.py
```

First run downloads the MIT-BIH records via WFDB, trains the model, and saves it. Every run after that skips straight to monitoring.

---

## Project Structure

```
ecg-ai-monitor/
├── ecg_monitor.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Industrial Applications

- Portable ECG monitoring devices and wearable diagnostics
- Edge AI healthcare monitoring platforms
- Real time biomedical signal analytics systems
- CPU based medical inference engines
- Intelligent monitoring interfaces for clinical environments

## Societal Applications

- Early detection of cardiac abnormalities
- Affordable cardiac monitoring in low resource regions
- Remote healthcare and telemedicine support
- Continuous monitoring for elderly populations
- Preventive healthcare through real time analytics

---

## Author

**Vaibhav Krishna V**  
Electronics & Communication Engineer  
📧 vaibhavkv078@gmail.com

---

## License

MIT — see [LICENSE](LICENSE) for details.
