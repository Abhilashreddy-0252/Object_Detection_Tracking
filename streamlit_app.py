import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from src.detector import YOLOv8Detector
from src.tracker import SortTracker
from src.visualizer import Visualizer

# ---- Page Config ----
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide"
)

# ---- Styling ----
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎯 Object Detection & Tracking</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by YOLOv8 + SORT Algorithm</p>', unsafe_allow_html=True)
st.divider()

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence = st.slider("Confidence Threshold", 0.10, 1.0, 0.30, 0.05)
    
    model_choice = st.selectbox("YOLO Model", ["yolov8n.pt (Fast - 80 classes)", "yolov8s.pt (Balanced - 80 classes)"])
    model_name = model_choice.split(" ")[0]
    
    max_age = st.slider("Tracker Max Age (frames)", 5, 100, 30)
    min_hits = st.slider("Tracker Min Hits", 1, 10, 3)
    
    st.divider()
    st.header("📋 Controls")
    st.markdown("""
    - **Upload** a video file (MP4, AVI, MOV)
    - **Capture** a photo from your webcam
    - Adjust **confidence** to filter detections
    - Tweak **tracker settings** in the sidebar
    """)
    
    st.divider()
    st.markdown("**Built with:** Python, OpenCV, YOLOv8, SORT")


# ---- Cache Model Loading ----
@st.cache_resource
def load_detector(model_name, confidence):
    model_path = model_name
    detector = YOLOv8Detector(model_path, confidence)
    return detector


def process_frame(frame, detector, tracker, visualizer, confidence):
    """Process a single frame: detect, track, visualize."""
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    frame = visualizer.draw_detections_and_tracks(frame, tracks)
    
    # Count objects by class
    class_counts = {}
    for track in tracks:
        cls_id = int(track[5])
        cls_name = detector.get_class_name(cls_id)
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    return frame, len(tracks), class_counts


# ---- Input Mode Selection ----
st.markdown("### 📹 Select Input Mode")
mode = st.radio("Choose input source:", ["📁 Upload Video File", "📸 Webcam Snapshot"], horizontal=True)

if mode == "📁 Upload Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Load model
        detector = load_detector(model_name, confidence)
        detector.confidence_threshold = confidence
        tracker = SortTracker(max_age=max_age, min_hits=min_hits)
        visualizer = Visualizer(detector.get_class_name)
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        st.markdown(f"**Video Info:** {total_frames} frames | {fps:.0f} FPS | Duration: {total_frames/fps:.1f}s")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            frame_placeholder = st.empty()
        
        with col2:
            metrics_placeholder = st.empty()
            class_placeholder = st.empty()
        
        # Controls
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            start = st.button("▶️ Start Detection", use_container_width=True)
        with col_b:
            skip_frames = st.number_input("Process every Nth frame", min_value=1, max_value=10, value=2)
        with col_c:
            st.info(f"Processing ~{total_frames // skip_frames} frames")
        
        if start:
            progress_bar = st.progress(0)
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % skip_frames != 0:
                    continue
                
                # Process
                processed_frame, num_tracks, class_counts = process_frame(
                    frame, detector, tracker, visualizer, confidence
                )
                
                # Calculate FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Display frame (BGR to RGB)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Display metrics
                metrics_placeholder.markdown(f"""
                <div class="metric-card">
                    <h2>📊 Live Stats</h2>
                    <p><strong>Tracked Objects:</strong> {num_tracks}</p>
                    <p><strong>Processing FPS:</strong> {current_fps:.1f}</p>
                    <p><strong>Frame:</strong> {frame_count}/{total_frames}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display class breakdown
                if class_counts:
                    breakdown = "\n".join([f"- **{name}**: {count}" for name, count in sorted(class_counts.items())])
                    class_placeholder.markdown(f"### 🏷️ Detected Classes\n{breakdown}")
                
                # Update progress
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            cap.release()
            progress_bar.progress(1.0)
            st.success(f"✅ Done! Processed {frame_count} frames in {elapsed:.1f}s ({current_fps:.1f} FPS)")
        
        # Cleanup
        try:
            os.unlink(tfile.name)
        except:
            pass

elif mode == "📸 Webcam Snapshot":
    st.markdown("Take a photo using your webcam and detect all objects in it!")
    
    camera_input = st.camera_input("📸 Capture a photo")
    
    if camera_input is not None:
        # Load model
        detector = load_detector(model_name, confidence)
        detector.confidence_threshold = confidence
        tracker = SortTracker(max_age=1, min_hits=1)
        visualizer = Visualizer(detector.get_class_name)
        
        # Read image
        file_bytes = np.frombuffer(camera_input.getvalue(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner("🔍 Detecting objects..."):
            processed_frame, num_tracks, class_counts = process_frame(
                frame, detector, tracker, visualizer, confidence
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_container_width=True, caption="Detected Objects")
        
        with col2:
            st.metric("Total Objects Detected", num_tracks)
            st.divider()
            if class_counts:
                st.markdown("### 🏷️ Breakdown")
                for name, count in sorted(class_counts.items()):
                    st.markdown(f"**{name}:** {count}")
            else:
                st.warning("No objects detected. Try lowering the confidence threshold.")

# ---- Footer ----
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    Object Detection & Tracking System | YOLOv8 + SORT | 
    <a href="https://github.com/Abhilashreddy-0252/Object_Detection_Tracking" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
