# 🫀 Clinical Grade Real-Time ECG AI Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A CPU-efficient, edge-ready cardiac monitoring pipeline combining biomedical signal processing with lightweight AI.*

</div>

---

## 📌 Project Overview

This project implements a **real-time ECG monitoring system** capable of:
- Streaming ECG signals in a sliding-window fashion
- Detecting R-peaks using adaptive thresholding
- Extracting physiological features (HR, HRV, RR intervals)
- Classifying cardiac rhythms using a lightweight deep learning model
- Validating predictions via hybrid AI + clinical rule logic
- Visualizing everything on a live dashboard

The system is designed for **edge deployment** — no GPU required, runs entirely on CPU.

---

## 🏗️ System Architecture

```
ECG Stream (MIT-BIH)
        ↓
  Bandpass Filter        ← Remove baseline wander & high-freq noise
        ↓
  R-Peak Detection       ← Adaptive thresholding on normalized signal
        ↓
  Feature Extraction     ← RR intervals, Heart Rate, HRV metrics
        ↓
  AI Classification      ← Lightweight CNN (CPU-optimized)
        ↓
  Hybrid Validation      ← AI + Tachycardia/Bradycardia clinical rules
        ↓
  Real-Time Dashboard    ← Live waveform, peaks, HR, classification
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Final Training Accuracy | **92.49%** |
| Final Training Loss | 0.2220 |
| Training Epochs | 6 |
| Inference Mode | CPU-only |
| Database | MIT-BIH Arrhythmia |

### Training Progression
| Epoch | Loss | Accuracy |
|---|---|---|
| 1/6 | 0.5961 | 80.57% |
| 2/6 | 0.3531 | 89.37% |
| 3/6 | 0.2914 | 91.11% |
| 4/6 | 0.2617 | 91.72% |
| 5/6 | 0.2383 | 92.20% |
| 6/6 | 0.2220 | **92.49%** |

---

## ✅ Key Features

- 🔴 **Real-time ECG streaming** from MIT-BIH Arrhythmia Database
- 🧠 **Hybrid Decision Logic** — AI predictions + deterministic clinical thresholds
- 📊 **Live PyQt5 dashboard** with waveform rendering, peak markers, HR display
- ⚡ **CPU-only inference** — deployable on low-power edge hardware
- 🏥 **Clinically relevant classification** — Normal, Tachycardia, Bradycardia, Arrhythmic

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Deep learning model training & inference |
| `NumPy` | Signal array manipulation |
| `SciPy` | Bandpass filtering (butter filter) |
| `WFDB` | MIT-BIH database access & streaming |
| `PyQtGraph` | High-performance waveform rendering |
| `PyQt5` | GUI dashboard framework |

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

### 3. Download MIT-BIH data (automatic via WFDB)
The script auto-downloads required records via the `wfdb` library on first run.

### 4. Run the system
```bash
python ecg_monitor.py
```

---

## 📁 Project Structure

```
ecg-ai-monitor/
├── ecg_monitor.py          # Main pipeline — run this
├── signal_processing.py    # Bandpass filter & R-peak detection
├── feature_extraction.py   # RR intervals, HR, HRV computation
├── model.py                # CNN classifier architecture
├── hybrid_validator.py     # AI + clinical rule hybrid logic
├── dashboard.py            # PyQt5 real-time visualization
├── requirements.txt        # All dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## 🏥 Applications

**Industrial:**
- Portable ECG monitoring devices and wearable diagnostics
- Edge AI healthcare monitoring platforms
- Real-time biomedical signal analytics systems
- CPU-based medical inference engines
- Intelligent monitoring interfaces for clinical environments

**Societal:**
- Early detection of cardiac abnormalities
- Affordable cardiac monitoring in low-resource regions
- Remote healthcare and telemedicine support
- Continuous monitoring for elderly populations
- Preventive healthcare through real-time analytics

---

## 👤 Author

**Vaibhav Krishna V**  
Electronics & Communication Engineering, NMIT Bengaluru  
USN: 1NT22EC182  
📧 vaibhavkv078@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
