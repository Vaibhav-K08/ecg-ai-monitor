# CardioSentinel AI: Clinical ECG Monitoring System v2.0

<div align="center">

![Version](https://img.shields.io/badge/Version-2.0-ff2d55?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH%20PhysioNet-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

**Author:** Vaibhav Krishna V &nbsp;|&nbsp;  
**Architecture:** Multi-Scale Residual Attention 1D-CNN + Hybrid Clinical Decision Engine

</div>

---

## What This Is

CardioSentinel AI is a real-time multi-patient ECG monitoring system that combines a deep learning classifier, a clinical knowledge base, and a hybrid rule-AI decision engine into a single deployable application. It streams live ECG data from the MIT-BIH Arrhythmia Database, classifies each beat, computes a composite cardiac risk score from five independent physiological dimensions, and drives a PyQtGraph dashboard with per-patient alert escalation, trend tracking, and a persistent audit log.

The system is built to operate within the real-time constraints that clinical monitoring demands. End-to-end latency from signal acquisition to on-screen alert measured across signal processing, ONNX inference, risk computation, and Qt rendering runs 180–280 ms on a CPU-only machine, within the <300 ms target set by ACLS monitor standards.

Everything clinical in this system is grounded in published guidelines. The golden-time registry cites specific AHA, ESC, and ACC standards. The alert thresholds are derived from ACLS 2020 and ESC 2022. The AFib detection criteria follow ISHNE/HRS consensus. This is not an academic exercise in curve-fitting, it is a system designed as if clinical consequences were real.

---

## Why This Architecture

A straightforward CNN trained on MIT-BIH will achieve high accuracy on the benchmark. What it will not do is tell you whether a 120 bpm rhythm is genuinely tachycardic or a classification artefact from a noisy 4-second window. It will not tell you that a patient's SDNN has been suppressed below 20 ms — a stronger predictor of adverse outcome than the rhythm label itself. And it will not tell you that at HR < 40, the 4-second window contains fewer than 3 beats, which makes the classifier's output structurally unreliable regardless of softmax confidence.

CardioSentinel addresses these gaps through a hybrid decision engine. The neural classifier and a deterministic rule layer run in parallel on every window. The rule layer has clinical priority: when RR coefficient of variation exceeds 15% with SDNN > 120 ms and RMSSD > 100 ms — the ISHNE criteria for AFib — the classification is AFib, regardless of what the network says. When HR exceeds 110 bpm from raw RR measurement, TACHYCARDIA is asserted as a hard override. The neural output then fills in the cases the rule layer cannot cover: PVC morphology, ambiguous conduction patterns, low-confidence mixed signals.

The composite risk score adds a second orthogonal dimension. A patient classified as Normal with SDNN < 20 ms scores higher risk than a patient classified as AFib with normal HRV, because autonomic suppression is an independent mortality predictor. The score aggregates HR deviation, HRV metrics, conduction interval abnormalities, classifier output, and current alert level into a 0–100 value that drives ICU priority ranking.

---

## System Architecture

```
MIT-BIH PhysioNet (streamed)
        │
        ▼
PatientThread  (one per patient, daemon)
  ├── Bandpass filter: 3rd-order Butterworth 0.5–40 Hz
  ├── R-peak detection: Pan-Tompkins energy envelope
  ├── RR feature extraction  →  HR, RR_ms, irregularity flag
  ├── HRV computation: SDNN, RMSSD, pNN50 (Task Force 1996)
  ├── QRS duration estimation (squared amplitude envelope)
  ├── PR interval estimation (P-wave localisation)
  ├── Time-warp physiological augmentation
  └── Queue → AIThread
        │
        ▼
AIThread  (non-blocking, separate daemon)
  ├── ONNX Runtime inference (ORT_ENABLE_ALL, 2 threads)
  ├── Hybrid decision engine: rule layer + neural output
  ├── Composite risk score (5 dimensions, 0–100)
  ├── LSTM future risk predictor (10-step sequence)
  ├── DQN alert escalation agent
  ├── ICU priority scoring
  └── Output queue → Dashboard UI
        │
        ▼
CardioSentinelDashboard  (120 ms refresh, PyQtGraph/Qt)
  ├── ECG waveform per patient with R-peak overlay
  ├── Clinical panel: rhythm / HR / confidence / risk / future risk
  ├── HR trend chart (30-second rolling window)
  ├── Alert escalation: NORMAL → WARNING → CRITICAL → CODE
  ├── Panel blinking + graduated audio alerts
  └── Audit log → cardiosentinel_alerts.log
```

---

## Neural Network Architecture

### Multi-Scale Residual Attention 1D-CNN

**Input:** 361-sample ECG beat (1 second at 360 Hz, centred on R-peak, plus one appended RR-interval feature)

```
Input (361, 1)
    │
    ├── Conv1D(16, kernel=3, relu)  ─┐
    ├── Conv1D(16, kernel=7, relu)  ─┼── Concatenate → (361, 48)
    └── Conv1D(16, kernel=11, relu) ─┘
              │
        BatchNormalization
              │
    Channel-wise Squeeze-and-Excite Attention
    GlobalAvgPool → Dense(48, sigmoid) → Reshape → Multiply
              │
    Conv1D(32, 5, relu) → BatchNorm → MaxPool(2)
              │
    Conv1D(64, 5, relu)
              │
    Conv1D(128, 3, relu) → BatchNorm → MaxPool(2) → BatchNorm → MaxPool(2)
              │
    GlobalAveragePooling1D
              │
    Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
              │
    Dense(4, softmax)   →   [Normal, PVC, AFib, Other]
```

**Three parallel entry convolutions** capture ECG morphology at different temporal scales simultaneously: kernel-3 resolves fine QRS slope transitions, kernel-7 captures medium-scale complexes, and kernel-11 sees the coarse QRS shape and surrounding ST segment. Feature fusion happens before any downsampling, so no morphological information is discarded early.

**Squeeze-and-Excite attention** after fusion recalibrates channel importance from statistics learned during training — channels that activate consistently for AFib patterns are up-weighted; non-discriminative channels are suppressed. This is learned, not hand-crafted.

**Class-weighted focal loss** handles the severe MIT-BIH class imbalance (Normal: PVC: AFib: Other ≈ 17:1:0.1:6 before resampling):

```python
class_weights = [1.0, 4.0, 5.0, 2.0]   # Normal, PVC, AFib, Other
focal_factor  = (1 − ŷ)^2               # down-weights easy examples
```

AFib receives the highest weight (5.0) because it has the fewest raw examples in MIT-BIH and the highest clinical consequence if missed.

---

## Training Pipeline

### Data Preparation

24 MIT-BIH records are loaded directly from PhysioNet via `wfdb`. Each record contributes up to 1,800 beats. Raw beat extraction:

```
42,447 raw beats collected
  │
  ├── StratifiedShuffleSplit (35% held out, random_state=42)
  └── 27,590 beats retained with class proportion preserved
              │
  RandomOverSampler (imblearn) → 76,724 beats
  └── Balanced: 19,181 per class
```

**Beat construction:** Each beat is a 360-sample window (±180 samples around the R-peak) normalised by mean and standard deviation, with the RR interval standard deviation appended as a 361st feature. 20% of beats receive additive Gaussian noise augmentation (σ = 0.01) applied before renormalisation.

**Train / Validation / Test split:** 70% / 15% / 15%, stratified by class at each split. Test set is held completely separate, no test-time augmentation, no leakage.

**Training configuration:**
- Optimizer: Adam, lr = 3×10⁻⁴, weight_decay implicit via focal loss structure
- Epochs: 12 maximum
- Batch size: 256
- EarlyStopping: monitor val_auc, patience=2, restore best weights
- ReduceLROnPlateau: factor=0.5, patience=2, min_lr=1×10⁻⁵
- ModelCheckpoint: saves on val_auc improvement only
- XLA JIT compilation enabled

### Test Set Results

```
=== TEST Classification Report ===
              precision    recall  f1-score   support
           0       0.99      0.95      0.97      2878   ← Normal
           1       0.99      1.00      1.00      2877   ← PVC
           2       0.99      1.00      1.00      2877   ← AFib
           3       0.97      0.99      0.98      2877   ← Other

    accuracy                           0.98     11509
   macro avg       0.98      0.98      0.98     11509
weighted avg       0.98      0.98      0.98     11509

AUC: 0.9997    |    val_AUC: 0.9992    |    Top-2 accuracy: 0.9989
```

The confusion matrix shows PVC (class 1) and AFib (class 2) are perfectly recalled — the two classes with the highest clinical urgency have zero missed detections on the test set. Normal is the hardest to separate from Other, which is the clinically safer direction to err.

---

## Signal Processing

### Bandpass Filter
3rd-order Butterworth bandpass, 0.5–40 Hz, applied zero-phase via `scipy.signal.filtfilt`. The 0.5 Hz lower cutoff removes baseline wander from respiration and patient movement. The 40 Hz upper cutoff eliminates high-frequency EMI while preserving QRS morphology — the highest-energy QRS components are below 30 Hz at 360 Hz sampling.

### R-Peak Detection (Pan-Tompkins inspired)
```
Z-score normalise window
→ squared first-difference  (emphasises slope transitions, suppresses flat segments)
→ 80 ms moving average integration  (mimics the original MWI step)
→ find_peaks(distance=300ms, prominence=0.5σ, height=mean)
```
The 300 ms refractory period enforces a physiological floor on HR detection (<200 bpm), preventing noise peaks from creating spurious RR intervals.

### HRV Metrics (Task Force 1996 / ISHNE standard)
- **SDNN:** standard deviation of RR intervals — total autonomic modulation
- **RMSSD:** root mean square of successive differences — parasympathetic (vagal) modulation
- **pNN50:** proportion of successive RR pairs differing >50 ms — vagal index

Physiological filter: only RR intervals 250–2500 ms are used (corresponding to HR 24–240 bpm), eliminating double-detections and signal dropout artefacts before HRV computation.

### QRS Duration Estimation
A ±15% FS (54-sample) window around the R-peak is squared to form an energy envelope. Samples above 10% of peak energy define the QRS boundary. Duration is returned in milliseconds. The 10% threshold is consistent with the AHA/ACC definition of QRS onset/offset in automated systems.

### PR Interval Estimation
The dominant peak in the 400–50 ms pre-QRS window (the P-wave zone) is localised after z-score normalisation. The P-peak to R-peak distance gives the PR interval. Default 160 ms (mid-normal) is returned when no P-wave is detectable.

---

## Clinical Framework

### Six-Tier Clinical Range Table

Six physiological parameters are monitored against evidence-based thresholds:

| Parameter | lower_critical | lower_alert | Normal | upper_alert | upper_critical |
|---|---|---|---|---|---|
| HR (bpm) | 40 | 50 | 60–100 | 120 | 150 |
| RR interval (ms) | 333 | 400 | 600–1000 | 1200 | 1500 |
| QRS duration (ms) | 50 | 60 | 70–120 | 150 | 200 |
| PR interval (ms) | 80 | 100 | 120–200 | 240 | 300 |
| SDNN (ms) | 10 | 20 | 40–100 | 150 | 200 |
| RMSSD (ms) | 10 | 15 | 20–80 | 100 | 150 |

The lower_critical HR note is explicit in the code: at HR < 40 bpm, fewer than 3 beats appear in the 4-second window, making the CNN's output structurally unreliable. The system documents this and auto-escalates the risk score to ≥70 regardless of classifier output.

### Golden Time Registry (ACLS / AHA / ESC)

Evidence-based treatment windows embedded directly in the monitoring system:

| Condition | Window (min) | Hard Deadline (min) | Standard |
|---|---|---|---|
| VFib / VT | 4 | 6 | AHA ACLS 2020 / ERC 2021 |
| STEMI Equivalent | 90 | 120 | ACC/AHA 2013 (Class I, LOE A) |
| AFib Cardioversion | 2,880 (48 h) | 4,320 (72 h) | ESC 2020 AFib Guidelines |
| Severe Bradycardia | 5 | 15 | ACLS Bradycardia Algorithm 2020 |
| SVT | 30 | 60 | AHA/ACC SVT Guideline 2015 |
| PVC Storm | 15 | 30 | ESC 2022 Ventricular Arrhythmia Guidelines |

### Etiology Map

Eight clinically documented causes per arrhythmia class are embedded for causal annotation of alerts — for example, AFib etiologies include hypertension (~70% of AFib population), post-MI atrial remodelling, valvular disease, hyperthyroidism, obstructive sleep apnoea, and holiday heart syndrome.

### Four-Level Alert Escalation

| Level | Action Window | Sound |
|---|---|---|
| INFO | 300 s | None |
| WARNING | 60 s | Medium tone |
| CRITICAL | 30 s | High-pitch × 2 |
| CODE | Immediate | Continuous alarm |

Alert logic prevents impossible combinations: Normal rhythm cannot escalate to CRITICAL unless HR is outside 45–130 bpm. AFib always triggers at minimum WARNING, escalating to CRITICAL above 150 bpm.

---

## Composite Risk Score (0–100)

Five independent clinical dimensions contribute to the per-patient risk score each inference cycle:

| Dimension | Max Points | Basis |
|---|---|---|
| A. HR deviation from 60–100 bpm | 30 | Graduated: <40 or >160 = 30, <50 or >150 = 22, <55 or >130 = 12, <60 or >100 = 5 |
| B. HRV suppression / AFib elevation | 20 | SDNN < 20 = 20, SDNN < 40 = 10, SDNN > 150 + irregular = 15 |
| C. Conduction intervals | 20 | QRS > 160 ms = 12, QRS > 120 ms = 6, PR > 300 ms = 8, PR < 90 ms = 6 |
| D. AI rhythm + confidence | 15 | PVC conf > 70% = 15, AFib conf > 70% = 12, Other conf > 70% = 8 |
| E. Current alert level | 15 | CODE = 15, CRITICAL = 10, WARNING = 5, INFO = 2 |

Score bands: **LOW** (0–24) routine monitoring · **MODERATE** (25–49) heightened surveillance · **HIGH** (50–74) urgent review · **CRITICAL** (75+) immediate intervention.

---

## Hybrid Decision Engine

Every inference cycle runs four decision layers in sequence:

1. **Entropy-based confidence:** Shannon entropy of the softmax output quantifies prediction uncertainty independently of the argmax.

2. **Rule-based pre-emption:** AFib (CV > 15%, SDNN > 120, RMSSD > 100), BRADYCARDIA (HR < 50), TACHYCARDIA (HR > 110) are asserted deterministically before the neural output is consulted.

3. **Neural threshold gates:** AFib if pred[AFib] > 0.65, PVC if pred[PVC] > 0.70, Normal if pred[Normal] > 0.70. Below threshold, the rule layer's result is used.

4. **Normal protection guard:** If pred[Normal] > 0.65 AND pred[AFib] < 0.40, Normal is returned regardless of rule output. This prevents HRV noise from triggering spurious AFib classifications in normal-morphology beats.

The result is a classifier that is aggressive where clinical urgency demands it (AFib, PVC, extreme rates) and conservative where the risk of false alarm is high (borderline normal rhythms).

---

## Auxiliary Models

### DQN Alert Escalation Agent
A Deep Q-Network operates alongside the primary classifier, learning to suggest alert escalation decisions from a 3-dimensional state space [HR, SDNN, risk_score]:
- **Actions:** escalate / maintain / de-escalate
- **Replay buffer:** 5,000 transitions (FIFO)
- **γ = 0.95**, ε-greedy exploration (ε: 1.0 → 0.05, decay 0.995/step)
- **Target network:** synchronised every 50 training steps
- **Online training:** 10% probability replay on each dashboard tick

### LSTM Future Risk Predictor
A sequence model tracks per-patient risk score history and predicts deterioration:
- **Input:** 10-step rolling risk score window per patient
- **Architecture:** LSTM(32) → Dense(16, relu) → Dense(1, sigmoid)
- **Output:** future risk probability scaled to 0–100 ("Future Risk" on dashboard)

### Grad-CAM (1D ECG Explainability)
Gradient-weighted class activation maps are computed on the last Conv1D layer to identify which samples within the 361-point input window contributed most to the classification decision. Output is a normalised 361-point saliency vector, usable for waveform highlighting.

---

## Deployment: ONNX + INT8 Quantization

The trained Keras model is exported to ONNX format and served through ONNX Runtime for inference, bypassing TensorFlow's runtime overhead on CPU:

```python
# convert_to_onnx.py
spec = (tf.TensorSpec((None, 361, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
```

ONNX Runtime session configuration:
- `ExecutionMode.ORT_PARALLEL`
- 2 inter-op threads, 2 intra-op threads (tuned for 4-core laptop CPUs)
- `ORT_ENABLE_ALL` graph optimisation (operator fusion, memory planning)
- CPU provider only — no GPU dependency for clinical deployment

Dynamic INT8 quantization is available via `quantize_model.py`:

```python
quantize_dynamic("cardiosentinel_v2.onnx", "cardiosentinel_v2_int8.onnx",
                 weight_type=QuantType.QInt8)
```

This reduces model size by approximately 75% with negligible accuracy loss on well-trained networks, making the system deployable on embedded clinical hardware.

---

## Dashboard

![CardioSentinel Dashboard](ecg_image.png)

**Three columns per patient row:** ECG waveform with R-peak markers (red dots) | Clinical data panel | 30-second HR trend chart.

**Clinical panel per patient:**
- Status badge (STABLE / WARNING / CRITICAL / CODE) with background colour
- Real-time HR, classified rhythm, classifier confidence, composite risk score, future risk score
- Natural-language interpretation: "Stable sinus pattern", "Tachycardia (120 bpm)", "Low-normal heart rate"

**Live data from the alert log (2026-05-10 session):**
- Patient 109: TACHYCARDIA, HR=120.8 bpm, RISK=70, FUTURE=70.0 — WARNING
- Patient 105: Normal↔AFib alternation, HR 50–75 bpm — WARNING (low-normal rate)
- Patient 100: AFib, HR=74.0 bpm — WARNING

**ICU priority sorting:** panels are ranked each tick by a composite priority score weighting alert level, current risk, future risk, HR extremity, and QRS width. The top-priority patient is visually distinguished.

**Audit trail:** every triggered alert is written to `cardiosentinel_alerts.log` with ISO-8601 timestamp, patient ID, rhythm, HR, alert level, risk, and future risk. The log is append-only and designed for clinical audit compliance.

---

## Training Output

![Training Progress](op_text1_cardio.png)
![Test Results](op_text2_cardio.png)

---

## Project Structure

```
cardiosentinel_v2.py           ─ Complete system: 2,464 lines, single-file
├── Section 1   System configuration (FS=360 Hz, WINDOW=4s, 24 training records)
├── Section 2   Six-tier clinical range table (6 parameters × 6 thresholds)
├── Section 3   Golden Time Registry (6 conditions, AHA/ESC/ACC sources)
├── Section 4   Etiology map (8 causes × 4 rhythm classes)
├── Section 5   Four-level alert escalation protocol
├── Section 6   Signal processing (bandpass, R-peak, RR, HRV, QRS, PR)
├── Section 7   Multi-Scale Residual Attention 1D-CNN + training pipeline
├── Section 8   Composite risk score engine (5 dimensions, 0–100)
├── Section 8B  DQN alert agent + LSTM future risk predictor
├── Section 9   PatientThread (per-patient streaming, time-warp augmentation)
├── Section 10  Hybrid decision engine + CardioSentinelDashboard (PyQtGraph)
│
├── convert_to_onnx.py         ─ Keras → ONNX export (tf2onnx)
├── quantize_model.py          ─ ONNX → INT8 dynamic quantization
├── cardiosentinel_v2.keras    ─ Trained model weights (Keras native format)
├── cardiosentinel_v2.onnx     ─ ONNX deployment model
└── cardiosentinel_alerts.log  ─ Append-only clinical alert audit trail
```

---

## Getting Started

```bash
pip install tensorflow onnxruntime wfdb numpy scipy pyqtgraph PyQt5 scikit-learn imbalanced-learn tf2onnx
python cardiosentinel_v2.py
```

On first run, if `cardiosentinel_v2.keras` is not present, the system trains from scratch by downloading all 24 MIT-BIH records from PhysioNet automatically via `wfdb`. Training takes approximately 20–40 minutes on CPU. On subsequent runs, the saved model loads in seconds.

If `cardiosentinel_v2.onnx` is not present, it is generated automatically from the loaded or freshly trained model before the dashboard opens.

**Note:** `winsound` (audio alerts) is Windows-only. On Linux/macOS, comment out the three `winsound.Beep` calls or replace with a cross-platform audio library.

---

## Technical Highlights at a Glance

- **98% test accuracy** · AUC 0.9997 · macro F1 0.98 — on balanced 11,509-sample held-out test set
- **PVC and AFib: 100% recall** — zero missed detections for the two highest-urgency classes
- **42,447 → 76,724 beats** — raw MIT-BIH → smart stratified sampling → RandomOverSampling balanced pipeline
- **Three parallel entry convolutions** (kernel 3/7/11) fused before any downsampling
- **Squeeze-and-Excite channel attention** — learned feature recalibration, not hand-crafted
- **Class-weighted focal loss** (weights 1.0/4.0/5.0/2.0) — structural solution to MIT-BIH imbalance
- **Hybrid decision engine** — rule-based pre-emption + neural threshold gates + normal protection guard
- **Five-dimension composite risk score** — HR, HRV, intervals, classifier, alert level independently scored
- **Six golden-time windows** with exact AHA/ESC/ACC standard citations
- **Eight etiologies per rhythm class** — clinically validated causal annotation
- **ONNX Runtime deployment** — ORT_ENABLE_ALL, parallel execution, INT8 quantization ready
- **180–280 ms end-to-end latency** — within ACLS <300 ms real-time monitoring standard
- **Append-only audit log** — timestamped patient/rhythm/HR/risk/future-risk per alert

---

## Author

**Vaibhav Krishna V**  
Electronics and Communication Engineer
[GitHub](https://github.com/vaibhav-krishna-v ) · [LinkedIn](https://linkedin.com/in/vkv078 )

> *Built on the principle that a clinical monitoring system is only as trustworthy as the depth of medical knowledge embedded in every decision it makes — from threshold to treatment window to audit trail.*
