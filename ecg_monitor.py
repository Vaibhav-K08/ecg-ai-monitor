"""
Clinical Grade Real Time ECG AI Monitoring System
Author: Vaibhav Krishna V
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import wfdb
import numpy as np
import tensorflow as tf
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, filtfilt, find_peaks
import threading, queue, time
from datetime import datetime

# ================= CONFIG =================
FS = 360
WINDOW_SEC = 4
WINDOW = FS * WINDOW_SEC
MODEL_PATH = "ecg_clinical_cpu.keras"
PATIENTS = ["100", "101"]

CLASSES = ["Normal", "PVC", "AFib", "Other"]

TACHY = 120
BRADY = 45

model_lock = threading.Lock()

# ================= FILTER =================
def bandpass(sig):
    nyq = 0.5 * FS
    b, a = butter(2, [0.5/nyq, 40/nyq], btype="band")
    return filtfilt(b, a, sig)

# ================= R PEAK DETECTION =================
def detect_rpeaks(ecg):
    s = (ecg - np.mean(ecg)) / (np.std(ecg)+1e-6)
    energy = s**2
    kernel = np.ones(int(0.08*FS))/(0.08*FS)
    energy = np.convolve(energy, kernel, mode="same")

    peaks,_ = find_peaks(
        energy,
        distance=int(0.3*FS),   # refractory period
        prominence=np.std(energy)
    )
    return peaks

# ================= CLINICAL HR =================
def compute_hr(peaks):
    if len(peaks) < 3:
        return 0, []

    rr = np.diff(peaks) / FS
    rr = rr[rr > 0.3]  # remove impossible beats

    if len(rr) < 2:
        return 0, []

    rr_med = np.median(rr)
    hr = 60 / rr_med
    hr = np.clip(hr, 30, 180)
    return hr, rr

# ================= HRV =================
def hrv(rr):
    if len(rr) < 3:
        return 0,0
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    return sdnn, rmssd

# ================= AFIB =================
def detect_afib(rr):
    sdnn, rmssd = hrv(rr)
    return sdnn > 0.12 and rmssd > 0.1

# ================= MODEL =================
def build_model():
    inp = tf.keras.Input(shape=(360,1))
    x = tf.keras.layers.Conv1D(32,5,activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(64,5,activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64,activation="relu")(x)
    out = tf.keras.layers.Dense(4,activation="softmax")(x)

    model = tf.keras.Model(inp,out)
    model.compile("adam","categorical_crossentropy",metrics=["accuracy"])
    return model

def map_label(sym):
    return 0 if sym=="N" else (1 if sym=="V" else (2 if sym=="A" else 3))

def load_or_train():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    print("Training clinical CPU model...")
    X,y = [],[]

    records = ["100","101","102","103","104","105","106"]

    for rec in records:
        r = wfdb.rdrecord(rec,pn_dir="mitdb")
        ann = wfdb.rdann(rec,"atr",pn_dir="mitdb")
        sig = r.p_signal[:,0]

        for i,p in enumerate(ann.sample):
            if p-180 < 0 or p+180 > len(sig): continue
            beat = sig[p-180:p+180]
            X.append(beat)
            y.append(map_label(ann.symbol[i]))

    X = np.array(X).reshape(-1,360,1)
    y = tf.keras.utils.to_categorical(y,4)

    model = build_model()
    model.fit(X,y,epochs=6,batch_size=128,verbose=1)
    model.save(MODEL_PATH)
    return model

model = load_or_train()

# ================= THREAD =================
class PatientThread(threading.Thread):
    def __init__(self, name, q):
        super().__init__()
        self.name = name
        self.q = q
        self.running = True

    def run(self):
        record = wfdb.rdrecord(self.name,pn_dir="mitdb",sampto=30000)
        sig = bandpass(record.p_signal[:,0])
        ptr = 0

        while self.running and ptr+WINDOW < len(sig):
            window = sig[ptr:ptr+WINDOW]

            peaks = detect_rpeaks(window)
            hr, rr = compute_hr(peaks)

            center = window[len(window)//2-180:len(window)//2+180]

            self.q.put((self.name, window, peaks, hr, rr, center))
            ptr += FS
            time.sleep(0.25)

# ================= DASHBOARD =================
class ECGDashboard:
    def __init__(self, patients):
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="🏥 Clinical ECG AI Monitor")
        self.win.resize(1500,900)
        self.win.show()

        self.queue = queue.Queue()
        self.curves = {}
        self.peaks = {}
        self.labels = {}
        self.threads = []

        for i,p in enumerate(patients):
            plot = self.win.addPlot(row=i,col=0,title=f"Patient {p}")
            plot.setYRange(-2,2)

            curve = plot.plot(pen=pg.mkPen("#00ffaa",width=2))
            peakp = plot.plot(symbol="o",pen=None,symbolBrush="r")

            self.curves[p] = curve
            self.peaks[p] = peakp

            label = pg.LabelItem(justify="left")
            self.win.addItem(label,row=i,col=1)
            self.labels[p] = label

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(150)

        for p in patients:
            t = PatientThread(p,self.queue)
            t.start()
            self.threads.append(t)

    def update(self):
        while not self.queue.empty():
            patient, window, peaks, hr, rr, center = self.queue.get()

            x = np.linspace(0,len(window)/FS,len(window))
            self.curves[patient].setData(x,window)

            if len(peaks):
                self.peaks[patient].setData(x[peaks],window[peaks])

            # ===== AI Inference =====
            with model_lock:
                pred = model.predict(center.reshape(1,360,1),verbose=0)[0]

            conf = np.max(pred)*100
            rhythm = CLASSES[np.argmax(pred)]

            # ===== Clinical Logic =====
            alarm = None

            if detect_afib(rr):
                alarm = "AFIB"

            if hr > TACHY and rhythm != "Normal":
                alarm = "TACHYCARDIA"

            if 0 < hr < BRADY:
                alarm = "BRADYCARDIA"

            # Color coding
            color = "#ff3333" if alarm else "#00ffaa"
            self.curves[patient].setPen(pg.mkPen(color,width=2))

            # ===== Label =====
            self.labels[patient].setText(f"""
            <span style='font-size:14pt'>
            HR: {hr:.1f} bpm<br>
            Rhythm: {rhythm}<br>
            Confidence: {conf:.1f}%<br>
            Status: {alarm if alarm else "Stable"}
            </span>
            """)

    def run(self):
        self.app.exec_()

# ================= MAIN =================
if __name__ == "__main__":
    dash = ECGDashboard(PATIENTS)
    dash.run()
