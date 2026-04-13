"""
app.py
------
Face Recognition Attendance System — Streamlit Application (v2.0)

Upgraded with:
  1. WebRTC camera (streamlit-webrtc) for stable live attendance
  2. SQLite database via SQLAlchemy
  3. Liveness detection (MediaPipe Face Mesh)
  4. Enhanced analytics, student profiles, PDF reports
"""

import io
import os
import sys
import time
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

# ── Project root on sys.path ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Deferred heavy imports ──────────────────────────────────────────────────
from utils.attendance import (
    mark_attendance,
    get_attendance_records,
    get_today_attendance,
    get_present_today,
    get_daily_counts,
    get_student_stats,
    get_weekly_summary,
)
from utils.dataset import (
    save_face_image,
    generate_embeddings_for_user,
    count_images,
    list_registered_users,
)
from utils.recognition import get_store, reload_embeddings, identify_face
from utils.reports import generate_pdf_report

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 18px;
        border: 1px solid rgba(99,179,237,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,179,237,0.15);
    }
    div[data-testid="metric-container"] label { color: #a0aec0 !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #63b3ed !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .page-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 24px;
        border-left: 5px solid #63b3ed;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .page-header h1 { color: #ffffff; margin: 0; font-size: 1.6rem; font-weight: 600; }
    .page-header p  { color: #a0aec0; margin: 4px 0 0; font-size: 0.9rem; }

    .badge-online  { background:#276749; color:#9ae6b4; padding:4px 14px;
                     border-radius:20px; font-size:0.8rem; font-weight:600;
                     display:inline-block; }
    .badge-offline { background:#742a2a; color:#feb2b2; padding:4px 14px;
                     border-radius:20px; font-size:0.8rem; font-weight:600;
                     display:inline-block; }
    .badge-liveness { background:#553c9a; color:#d6bcfa; padding:4px 14px;
                      border-radius:20px; font-size:0.8rem; font-weight:600;
                      display:inline-block; margin-top:4px; }

    .info-box {
        background:rgba(99,179,237,0.08); border:1px solid rgba(99,179,237,0.3);
        border-radius:12px; padding:16px 20px; margin-top:10px;
    }

    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(99,179,237,0.15);
        text-align: center;
        box-shadow: 0 2px 16px rgba(0,0,0,0.2);
    }
    .stat-card h3 { color: #63b3ed; margin: 0; font-size: 2rem; font-weight:700; }
    .stat-card p { color: #a0aec0; margin: 4px 0 0; font-size: 0.85rem; }

    .profile-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(99,179,237,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .profile-card h2 { color:#fff; margin:0 0 4px; font-weight:600; }
    .profile-card .subtitle { color:#a0aec0; font-size:0.85rem; }

    .liveness-bar {
        height: 6px;
        border-radius: 3px;
        background: #2d3748;
        overflow: hidden;
        margin: 8px 0;
    }
    .liveness-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "camera_running": False,
        "attendance_log": [],
        "last_recognized": {},
        "liveness_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_model_available():
    """Return (available: bool, message: str)."""
    model_path = os.path.join(BASE_DIR, "models", "arcface.onnx")
    if not os.path.exists(model_path):
        return False, (
            "❌ **ArcFace model not found.**  \n"
            "Run `python download_model.py` in the project directory."
        )
    return True, ""


def process_frame(frame, threshold):
    """
    Detect faces, run ArcFace, return annotated frame and list of (name, conf).
    """
    from utils.face_detection import get_detector
    from utils.arcface import get_embedding as arcface_embed

    detector = get_detector()
    boxes = detector.detect(frame)
    results = []
    labels = []

    for box in boxes:
        x, y, w, h = box
        face_crop = frame[
            max(0, y): min(frame.shape[0], y + h),
            max(0, x): min(frame.shape[1], x + w),
        ]
        if face_crop.size == 0:
            labels.append("Unknown (0.00)")
            results.append(("Unknown", 0.0))
            continue
        try:
            emb = arcface_embed(face_crop)
            name, conf = identify_face(emb, threshold=threshold)
        except Exception:
            name, conf = "Error", 0.0
        labels.append("{} ({:.2f})".format(name, conf))
        results.append((name, conf))

    annotated = detector.draw_results(frame, boxes, labels)
    return annotated, boxes, results


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
model_ok, model_msg = check_model_available()

with st.sidebar:
    st.markdown("## 🎓 Face Attendance")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📷 Live Attendance",
            "👨‍🎓 Add Student",
            "👤 Student Profiles",
            "📊 Records",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if model_ok:
        st.markdown('<span class="badge-online">● System Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">● Model Missing</span>', unsafe_allow_html=True)
        st.caption("Run `python download_model.py`")

    # Liveness toggle
    st.markdown("---")
    st.session_state.liveness_enabled = st.toggle(
        "🛡️ Liveness Detection", value=st.session_state.liveness_enabled
    )
    if st.session_state.liveness_enabled:
        st.markdown('<span class="badge-liveness">🛡️ Anti-Spoof ON</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("📅 {}".format(datetime.now().strftime("%A, %d %B %Y")))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="page-header"><h1>🏠 Dashboard</h1>'
        '<p>Real-time overview of the attendance system</p></div>',
        unsafe_allow_html=True,
    )

    registered = list_registered_users()
    present_today = get_present_today()
    all_records = get_attendance_records()
    daily_counts = get_daily_counts()

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Registered Users", len(registered))
    c2.metric("✅ Present Today", len(present_today))

    attendance_rate = 0
    if len(registered) > 0:
        attendance_rate = int(len(present_today) / len(registered) * 100)
    c3.metric("📈 Attendance Rate", f"{attendance_rate}%")
    c4.metric("⚙️ System Status", "🟢 Online" if model_ok else "🔴 Offline")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Daily Attendance Trend")
        if daily_counts:
            df_chart = pd.DataFrame(
                {"Date": list(daily_counts.keys()), "Count": list(daily_counts.values())}
            )
            bar_chart = alt.Chart(df_chart).mark_bar(
                color="#8b5cf6",
                cornerRadiusEnd=5,
                size=20
            ).encode(
                x=alt.X("Date:O", axis=alt.Axis(title="", labelAngle=-45, grid=False, domain=False, ticks=False)),
                y=alt.Y("Count:Q", axis=alt.Axis(title="", grid=True, domain=False, ticks=False)),
                tooltip=["Date", "Count"]
            ).configure_view(
                strokeWidth=0
            ).properties(
                height=300
            )
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No attendance data yet.")

    with col_right:
        st.subheader("🎯 Present Today")
        total_registered = len(registered)
        present_count = len(present_today)
        absent_count = total_registered - present_count

        if total_registered > 0:
            source = pd.DataFrame({
                "Status": ["Present", "Absent"],
                "Count": [present_count, absent_count]
            })
            base = alt.Chart(source).encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color("Status:N", scale=alt.Scale(
                    domain=["Present", "Absent"],
                    range=["#63b3ed", "#f87979"]
                ), legend=None),
                tooltip=["Status", "Count"]
            )
            donut = base.mark_arc(innerRadius=70, cornerRadius=2)
            text1 = alt.Chart(pd.DataFrame({"text": [f"{total_registered}"]})).mark_text(
                align="center", baseline="middle", fontSize=32, color="#e0e0e0",
                dy=-10, fontWeight="bold"
            ).encode(text="text:N")
            text2 = alt.Chart(pd.DataFrame({"text": ["TOTAL"]})).mark_text(
                align="center", baseline="middle", fontSize=14, color="#a0aec0", dy=20
            ).encode(text="text:N")
            st.altair_chart(
                (donut + text1 + text2).configure_view(strokeWidth=0),
                use_container_width=True
            )
            with st.expander("View Checked In Users"):
                if present_count:
                    for name in present_today:
                        st.markdown("✅ **{}**".format(name))
                else:
                    st.info("No attendance marked today.")
        else:
            st.info("No registered users yet.")

    # ── Weekly Heatmap ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Attendance Heatmap")
    if all_records:
        heatmap_data = []
        for rec in all_records:
            d = datetime.strptime(rec["date"], "%Y-%m-%d")
            heatmap_data.append({
                "Day": d.strftime("%a"),
                "Week": f"W{d.isocalendar()[1]}",
                "Count": 1,
                "Date": rec["date"],
            })
        df_heat = pd.DataFrame(heatmap_data)
        df_heat = df_heat.groupby(["Week", "Day"], as_index=False).agg({"Count": "sum"})

        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        heatmap = alt.Chart(df_heat).mark_rect(cornerRadius=4).encode(
            x=alt.X("Week:O", axis=alt.Axis(title="", labelAngle=0)),
            y=alt.Y("Day:O", sort=day_order, axis=alt.Axis(title="")),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="purples"), legend=None),
            tooltip=["Week", "Day", "Count"]
        ).configure_view(strokeWidth=0).properties(height=200)
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("No data for heatmap yet.")

    # ── Attendance by Student ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("👥 Attendance by Student")
    if all_records:
        student_counts = {}
        for rec in all_records:
            student_counts[rec["name"]] = student_counts.get(rec["name"], 0) + 1
        df_students = pd.DataFrame(
            {"Name": list(student_counts.keys()), "Days": list(student_counts.values())}
        ).sort_values("Days", ascending=True)

        student_bar = alt.Chart(df_students).mark_bar(
            cornerRadiusEnd=5, color="#63b3ed"
        ).encode(
            x=alt.X("Days:Q", axis=alt.Axis(title="Total Days Present")),
            y=alt.Y("Name:N", sort="-x", axis=alt.Axis(title="")),
            tooltip=["Name", "Days"]
        ).configure_view(strokeWidth=0).properties(height=max(150, len(student_counts) * 40))
        st.altair_chart(student_bar, use_container_width=True)
    else:
        st.info("No student data yet.")

    # ── Recent Attendance ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Recent Attendance (last 10)")
    if all_records:
        df_recent = pd.DataFrame(all_records[-10:][::-1])
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.info("No records found.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Live Attendance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📷 Live Attendance":
    st.markdown(
        '<div class="page-header"><h1>📷 Live Attendance</h1>'
        '<p>Real-time face recognition — attendance marked automatically</p></div>',
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error(model_msg)
        st.stop()

    # ── Try WebRTC first, fallback to OpenCV ─────────────────────────────────
    _webrtc_available = False
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        _webrtc_available = True
    except ImportError:
        pass

    # ── Controls ──────────────────────────────────────────────────────────────
    col_thresh, col_info = st.columns([1, 2])

    with col_thresh:
        threshold = st.slider(
            "Recognition Threshold",
            min_value=0.20, max_value=0.90, value=0.45, step=0.05,
        )

    with col_info:
        store = get_store()
        users = store.list_users()
        if users:
            st.markdown(
                '<div class="info-box">📂 {} user(s) loaded: {}</div>'.format(
                    len(users), ", ".join(users[:5]) + ("..." if len(users) > 5 else "")
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ No embeddings found. Add a student first.")

    if st.session_state.liveness_enabled:
        st.info("🛡️ **Liveness Detection is ON** — Students must blink and move slightly for verification.")

    st.markdown("---")

    if _webrtc_available:
        # ── WebRTC Approach ──────────────────────────────────────────────────
        class FaceRecognitionProcessor(VideoProcessorBase):
            def __init__(self):
                self._threshold = threshold
                self._lock = threading.Lock()
                self._attendance_log = []
                self._last_recognized = {}
                self._cooldown = 5
                self._liveness_checker = None
                self._liveness_enabled = st.session_state.get("liveness_enabled", True)
                self._liveness_state = {}  # per-person liveness tracking

                # Pre-load recognition data
                reload_embeddings()

                if self._liveness_enabled:
                    try:
                        from utils.liveness import LivenessChecker
                        self._liveness_checker = LivenessChecker()
                    except Exception:
                        self._liveness_checker = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)

                annotated, boxes, results = process_frame(img, self._threshold)

                # Liveness check + auto-mark
                for i, (name, conf) in enumerate(results):
                    if name not in ("Unknown", "Error"):
                        can_mark = True

                        # Liveness gate
                        if self._liveness_checker and boxes:
                            liveness = self._liveness_checker.update(img, boxes[i] if i < len(boxes) else None)
                            status = liveness.get("status", "checking")
                            msg = liveness.get("message", "")

                            if status == "checking":
                                can_mark = False
                                # Draw liveness progress on frame
                                progress = liveness.get("progress", 0)
                                bar_w = int(200 * progress)
                                bx = boxes[i][0] if i < len(boxes) else 10
                                by = boxes[i][1] + boxes[i][3] + 20 if i < len(boxes) else 30
                                cv2.rectangle(annotated, (bx, by), (bx + 200, by + 8), (45, 55, 72), -1)
                                cv2.rectangle(annotated, (bx, by), (bx + bar_w, by + 8), (139, 92, 246), -1)
                                cv2.putText(annotated, "Verifying...", (bx, by - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (139, 92, 246), 1)
                            elif status == "spoof":
                                can_mark = False
                                if i < len(boxes):
                                    bx, by = boxes[i][0], boxes[i][1] + boxes[i][3] + 20
                                    cv2.putText(annotated, "SPOOF DETECTED", (bx, by),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            elif status == "live":
                                if i < len(boxes):
                                    bx, by = boxes[i][0], boxes[i][1] + boxes[i][3] + 20
                                    cv2.putText(annotated, "LIVE", (bx, by),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if can_mark:
                            with self._lock:
                                last = self._last_recognized.get(name, 0)
                                if time.time() - last > self._cooldown:
                                    marked, msg = mark_attendance(name, conf)
                                    if marked:
                                        self._attendance_log.insert(0, msg)
                                        self._last_recognized[name] = time.time()
                                        # Reset liveness for next person
                                        if self._liveness_checker:
                                            self._liveness_checker.reset()

                return frame.from_ndarray(annotated, format="bgr24")

        ctx = webrtc_streamer(
            key="face-attendance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": {"width": 960, "height": 540}, "audio": False},
            async_processing=True,
        )

        # Show attendance log
        if ctx.video_processor:
            st.markdown("---")
            st.subheader("📝 Session Log")
            log_placeholder = st.empty()
            if hasattr(ctx.video_processor, '_attendance_log') and ctx.video_processor._attendance_log:
                for entry in ctx.video_processor._attendance_log[:10]:
                    st.markdown(entry)
            else:
                st.info("Waiting for recognitions…")
    else:
        # ── OpenCV Fallback ──────────────────────────────────────────────────
        st.warning("📦 `streamlit-webrtc` not available. Using OpenCV fallback (may freeze on stop).")

        col_btn = st.columns([1, 3])[0]
        with col_btn:
            if not st.session_state.camera_running:
                if st.button("▶️ Start Camera", use_container_width=True, type="primary"):
                    reload_embeddings()
                    st.session_state.camera_running = True
                    st.session_state.attendance_log = []
                    st.rerun()
            else:
                if st.button("⏹️ Stop Camera", use_container_width=True):
                    st.session_state.camera_running = False
                    st.rerun()

        if st.session_state.camera_running:
            frame_ph = st.empty()
            status_ph = st.empty()
            log_ph = st.empty()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam.")
                st.session_state.camera_running = False
                st.stop()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            COOLDOWN = 5

            try:
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        status_ph.warning("⚠️ Frame read failed — retrying…")
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    annotated, boxes, results = process_frame(frame, threshold)

                    for name, conf in results:
                        if name not in ("Unknown", "Error"):
                            last = st.session_state.last_recognized.get(name, 0)
                            if time.time() - last > COOLDOWN:
                                marked, msg = mark_attendance(name, conf)
                                if marked:
                                    st.session_state.attendance_log.insert(0, msg)
                                    st.session_state.last_recognized[name] = time.time()

                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_ph.image(annotated_rgb, use_column_width=True)

                    if boxes:
                        status_ph.success("👁️ {} face(s) detected".format(len(boxes)))
                    else:
                        status_ph.info("👁️ Waiting for faces…")

                    if st.session_state.attendance_log:
                        log_ph.markdown(
                            "**Recent markings:**\n" +
                            "\n".join(st.session_state.attendance_log[:5])
                        )
                    time.sleep(0.05)
            except Exception as e:
                if type(e).__name__ in ("RerunException", "StopException"):
                    raise
                st.error("Camera error: {}".format(e))
            finally:
                cap.release()
                st.session_state.camera_running = False
        else:
            st.markdown(
                '<div class="info-box" style="text-align:center; padding:40px;">'
                "📷 Click <b>Start Camera</b> to begin face recognition."
                "</div>",
                unsafe_allow_html=True,
            )
            if st.session_state.attendance_log:
                st.subheader("Last Session Log")
                for entry in st.session_state.attendance_log[:10]:
                    st.markdown(entry)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Add Student
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👨‍🎓 Add Student":
    st.markdown(
        '<div class="page-header"><h1>👨‍🎓 Add Student</h1>'
        '<p>Use webcam to capture face images and generate ArcFace embeddings</p></div>',
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error(model_msg)
        st.stop()

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.subheader("📝 Student Details")
        student_name = st.text_input(
            "Student Name",
            placeholder="e.g. Alice Johnson",
        )

        if student_name:
            img_count = count_images(student_name)
            if img_count > 0:
                st.info("📸 Images captured: **{}**".format(img_count))
            else:
                st.warning("No images captured yet. Use the camera on the right.")

            st.markdown("---")
            st.markdown("**Step 2 – Generate Embeddings**")
            st.caption("Requires at least 3 captured images.")

            gen_disabled = img_count < 3
            if st.button(
                "⚙️ Generate Embeddings",
                disabled=gen_disabled,
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("Running ArcFace on {} image(s)…".format(img_count)):
                    success, msg = generate_embeddings_for_user(student_name)
                    reload_embeddings()
                if success:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)

            if gen_disabled:
                st.caption("⬆️ Capture at least 3 images first.")
        else:
            st.info("👆 Enter a student name to begin.")

    with col_cam:
        st.subheader("📷 Capture Images")

        if student_name:
            st.caption(
                "Click the snapshot button (📷) inside the camera widget. "
                "Repeat 5–10 times from different angles."
            )

            cam_img = st.camera_input("Take a photo", key="add_student_cam")

            if cam_img is not None:
                from utils.face_detection import get_detector
                file_bytes = np.frombuffer(cam_img.getvalue(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is not None:
                    detector = get_detector()
                    boxes = detector.detect(frame)

                    if boxes:
                        filepath = save_face_image(student_name, frame, boxes[0])
                        new_count = count_images(student_name)
                        st.success(
                            "✅ Image {} saved! (Total: {})".format(
                                new_count, new_count
                            )
                        )
                    else:
                        st.warning("⚠️ No face detected in this image. Try again.")
                else:
                    st.error("Could not decode image.")
        else:
            st.info("Enter a student name first.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Student Profiles
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👤 Student Profiles":
    st.markdown(
        '<div class="page-header"><h1>👤 Student Profiles</h1>'
        '<p>Individual attendance history, streaks, and statistics</p></div>',
        unsafe_allow_html=True,
    )

    registered = list_registered_users()

    if not registered:
        st.info("No registered students yet. Add a student first.")
        st.stop()

    selected = st.selectbox("Select a student", registered, index=0)

    if selected:
        stats = get_student_stats(selected)
        history = get_attendance_records(name_filter=selected)

        # ── Profile Card ─────────────────────────────────────────────────────
        st.markdown(
            f'<div class="profile-card">'
            f'<h2>👤 {selected}</h2>'
            f'<p class="subtitle">Registered Student</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # ── Stats Row ────────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("📅 Total Days", stats["total_days"])
        s2.metric("🔥 Current Streak", f"{stats['streak']} days")
        s3.metric("🎯 Avg Confidence", f"{stats['avg_confidence']:.1%}")
        s4.metric("📸 Images", count_images(selected))

        # ── Date info ────────────────────────────────────────────────────────
        if stats["first_seen"]:
            st.markdown(
                f'<div class="info-box">'
                f'📍 First seen: <b>{stats["first_seen"]}</b> &nbsp;|&nbsp; '
                f'Last seen: <b>{stats["last_seen"]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Attendance Rate Ring ─────────────────────────────────────────────
        col_chart, col_history = st.columns([1, 2])

        with col_chart:
            st.subheader("📊 Attendance Rate")
            total_days_possible = len(get_daily_counts())
            if total_days_possible > 0:
                rate = stats["total_days"] / total_days_possible
                ring_data = pd.DataFrame({
                    "Type": ["Present", "Absent"],
                    "Value": [stats["total_days"], total_days_possible - stats["total_days"]]
                })
                ring = alt.Chart(ring_data).mark_arc(innerRadius=50, cornerRadius=3).encode(
                    theta=alt.Theta("Value:Q"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Present", "Absent"],
                        range=["#8b5cf6", "#2d3748"]
                    ), legend=None),
                    tooltip=["Type", "Value"]
                )
                ring_text = alt.Chart(pd.DataFrame({"text": [f"{rate:.0%}"]})).mark_text(
                    fontSize=28, color="#8b5cf6", fontWeight="bold"
                ).encode(text="text:N")
                st.altair_chart(
                    (ring + ring_text).configure_view(strokeWidth=0),
                    use_container_width=True
                )
            else:
                st.info("No data yet.")

        with col_history:
            st.subheader("📋 Attendance History")
            if history:
                df_hist = pd.DataFrame(history[::-1])
                df_hist.columns = [c.capitalize() for c in df_hist.columns]
                if "Confidence" in df_hist.columns:
                    df_hist["Confidence"] = df_hist["Confidence"].apply(
                        lambda x: "{:.1%}".format(float(x))
                    )
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
            else:
                st.info("No attendance records for this student.")

        # ── Danger Zone: Remove Student ──────────────────────────────────────
        st.markdown("---")
        with st.expander("⚠️ Danger Zone — Remove Student", expanded=False):
            st.warning(
                f"This will **permanently delete** all data for **{selected}**: "
                "images, embeddings, and attendance records. This action cannot be undone."
            )
            confirm_name = st.text_input(
                f"Type **{selected}** to confirm deletion",
                placeholder=f"Type '{selected}' here",
                key="delete_confirm",
            )
            if st.button("🗑️ Delete Student", type="primary", use_container_width=True):
                if confirm_name == selected:
                    from utils.dataset import delete_student
                    from utils.attendance import delete_student_records

                    # Delete attendance records from DB
                    deleted_records = delete_student_records(selected)

                    # Delete images + embeddings
                    success, msg = delete_student(selected)
                    reload_embeddings()

                    if success:
                        st.success(f"{msg} ({deleted_records} attendance record(s) removed)")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error(f"Name doesn't match. Please type **{selected}** exactly to confirm.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Records
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Records":
    st.markdown(
        '<div class="page-header"><h1>📊 Attendance Records</h1>'
        '<p>Browse, filter and export attendance data</p></div>',
        unsafe_allow_html=True,
    )

    all_records = get_attendance_records()

    # ── Filters ──────────────────────────────────────────────────────────────
    st.subheader("🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        name_filter = st.text_input("Filter by Name", placeholder="e.g. Alice")
    with f2:
        all_dates = sorted({r["date"] for r in all_records}, reverse=True)
        date_options = ["All Dates"] + all_dates
        date_sel = st.selectbox("Filter by Date", date_options)
        date_filter = None if date_sel == "All Dates" else date_sel
    with f3:
        date_from = st.date_input("From Date", value=None, key="date_from")
    with f4:
        date_to = st.date_input("To Date", value=None, key="date_to")

    # Convert date objects to strings
    date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
    date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None

    filtered = get_attendance_records(
        name_filter=name_filter or None,
        date_filter=date_filter,
        date_from=date_from_str,
        date_to=date_to_str,
    )

    # ── Summary metrics ──────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("📋 Filtered Records", len(filtered))
    s2.metric("👥 Unique Names", len({r["name"] for r in filtered}))
    s3.metric("📅 Date Range", len({r["date"] for r in filtered}))

    st.markdown("---")

    if filtered:
        df = pd.DataFrame(filtered[::-1])
        df.columns = [c.capitalize() for c in df.columns]
        df["Confidence"] = df["Confidence"].apply(lambda x: "{:.1%}".format(float(x)))

        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Export buttons ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📥 Export")
        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name="attendance_{}.csv".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with exp2:
            if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
                with st.spinner("Generating PDF…"):
                    pdf_bytes = generate_pdf_report(
                        name_filter=name_filter or None,
                        date_from=date_from_str,
                        date_to=date_to_str,
                    )
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="attendance_report_{}.pdf".format(
                        datetime.now().strftime("%Y%m%d_%H%M%S")
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )

        with exp3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    else:
        st.info("No records match the current filters.")
