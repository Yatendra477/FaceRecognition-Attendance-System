# 🎓 Face Recognition Attendance System v2.0

A real-time, AI-powered attendance system built with **Python**, **Streamlit**, and **ArcFace (ONNX)**.  
Features **WebRTC live camera**, **SQLite database**, **liveness detection**, and **PDF reporting**.

---

## 🗂️ Project Structure

```
FaceRecognition-Attendance-System/
├── app.py                   # Main Streamlit application (v2.0)
├── download_model.py        # ArcFace ONNX model downloader
├── requirements.txt
├── models/
│   └── arcface.onnx         # Downloaded ArcFace model (r50, ~166 MB)
├── dataset/
│   └── <StudentName>/       # Face images per student
├── embeddings/
│   └── embeddings.json      # ArcFace embedding vectors
├── data/
│   └── attendance.db        # SQLite attendance database
└── utils/
    ├── __init__.py
    ├── face_detection.py    # OpenCV DNN / Haar Cascade detector
    ├── arcface.py           # ONNX Runtime inference wrapper
    ├── recognition.py       # Cosine similarity + identity lookup
    ├── dataset.py           # Image capture & embedding generation
    ├── attendance.py        # Attendance queries (SQLAlchemy)
    ├── database.py          # SQLAlchemy models & engine
    ├── liveness.py          # Anti-spoofing (MediaPipe Face Mesh)
    └── reports.py           # PDF report generation (fpdf2)
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the ArcFace model

```bash
python download_model.py
```

This downloads `w600k_r50.onnx` (~166 MB) and saves it as `models/arcface.onnx`.

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPIs, daily trend bar chart, donut chart, attendance heatmap, per-student chart |
| 📷 Live Attendance | WebRTC camera feed with bounding boxes, auto-marks attendance, liveness gate |
| 👨‍🎓 Add Student | Capture face images → generate & store ArcFace embeddings |
| 👤 Student Profiles | Individual attendance history, streaks, attendance rate ring, stats |
| 📊 Records | Browse / filter / export attendance as CSV or PDF report |

---

## 🔄 Workflow

1. **Add Student** → enter name → capture ≥5 face images → Generate Embeddings  
2. **Live Attendance** → camera starts via WebRTC → faces detected & recognised → liveness verified → attendance marked automatically  
3. **Student Profiles** → view individual stats, streaks, attendance rates  
4. **Records** → filter by name/date/range → download CSV or generate branded PDF report  

---

## 🧠 Technical Details

| Component | Implementation |
|---|---|
| Face Detection | OpenCV DNN SSD (Caffe) with Haar Cascade fallback |
| Face Embeddings | ArcFace r50 via ONNX Runtime (512-dim) |
| Similarity | Cosine similarity (L2-normalised vectors → dot product) |
| Storage | SQLite via SQLAlchemy (auto-migrated from JSON) |
| Camera | streamlit-webrtc (WebRTC) with OpenCV fallback |
| Liveness | MediaPipe Face Mesh — Blink (EAR), Head Pose, Texture Analysis |
| Reports | fpdf2 — branded PDF with summary + attendance table |
| Charts | Altair — bar charts, donut charts, heatmaps |
| UI | Streamlit with Inter font, dark gradient theme |

### Recognition threshold
Default is **0.45** — adjust via the slider in the Live Attendance page.  
- Increase (e.g. 0.60) → fewer false positives  
- Decrease (e.g. 0.35) → fewer false negatives  

### Liveness Detection
Toggleable via the sidebar switch (default: ON).  
- **Blink Detection**: Eye Aspect Ratio (EAR) using 468 facial landmarks  
- **Head Pose**: Tracks yaw variation via cv2.solvePnP  
- **Texture Analysis**: Laplacian variance to detect flat photos/screens  
- Requires ≥1 blink and slight head movement within a 4-second window  

### Low-light handling
The Haar Cascade fallback applies histogram equalisation before detection.  
For best results, ensure adequate lighting when capturing dataset images.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `Model not found` | Run `python download_model.py` |
| `Cannot open webcam` | Check camera is not used by another app; try index 1 in `cv2.VideoCapture(1)` |
| Low recognition accuracy | Capture more images (10+) from different angles / lighting |
| App is slow | Reduce webcam resolution in `app.py` |
| WebRTC not working | Make sure `streamlit-webrtc` is installed; uses OpenCV fallback automatically |
| Liveness too strict | Toggle it off in the sidebar, or adjust thresholds in `utils/liveness.py` |
# FaceRecognition-Attendance-System
