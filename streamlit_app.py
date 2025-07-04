import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
from insightface.app import FaceAnalysis
import zipfile
from io import BytesIO

# Initialize InsightFace model
@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

app = load_model()

st.title("🎯 Multi-Video Multi-Face CCTV Matching")

# Upload multiple face images and videos
query_img_files = st.file_uploader("Upload One or More Query Face Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
video_files = st.file_uploader("Upload One or More CCTV Videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)

# Set similarity threshold
similarity_threshold = st.slider("Set Similarity Threshold", 0.2, 0.6, 0.28, 0.01)

# Choose scan speed
scan_option = st.radio("Choose Scan Speed", ["Slow Scan", "Fast Scan", "Faster Scan", "Fastest Scan"])
scan_map = {"Slow Scan": 5, "Fast Scan": 30, "Faster Scan": 60, "Fastest Scan": 120}
frame_skip = scan_map[scan_option]

# Session state setup
for key, default in {
    'scan_started': False,
    'scan_paused': False,
    'scan_stopped': False,
    'current_frame_num': 0,
    'all_matches': [],
    'match_images': [],
    'scan_complete': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Control buttons
col1, col2, col3 = st.columns(3)
if col1.button("🚀 Start Scan"):
    st.session_state.scan_started = True
    st.session_state.scan_paused = False
    st.session_state.scan_stopped = False
    st.session_state.scan_complete = False
    st.session_state.all_matches = []
    st.session_state.match_images = []
if col2.button("⏸️ Pause/Continue"):
    st.session_state.scan_paused = not st.session_state.scan_paused
if col3.button("🛑 Stop Scan"):
    st.session_state.scan_stopped = True
    st.session_state.scan_started = False

# Process if started
if query_img_files and video_files and st.session_state.scan_started:
    query_embeddings = []
    for img_file in query_img_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(img_file.read())
            img_path = tmp_img.name
        img = cv2.imread(img_path)
        faces = app.get(img)
        if faces:
            largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            embedding = largest_face.embedding / np.linalg.norm(largest_face.embedding)
            query_embeddings.append((img_file.name, embedding))

    if not query_embeddings:
        st.error("❌ No valid faces found in any query images.")
        st.stop()

    for video_file in video_files:
        if st.session_state.scan_stopped:
            break

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir = tempfile.mkdtemp()
        st.markdown(f"### 🔍 Processing video: {video_file.name}")
        progress = st.empty()
        current_frame_display = st.empty()
        df_placeholder = st.empty()

        def is_blurry(image, threshold=80):
            return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

        processed_frames = 0
        total_frames_to_process = total_frames // frame_skip
        start_time = datetime.datetime.now()

        frame_num = st.session_state.current_frame_num
        while frame_num < total_frames:
            if st.session_state.scan_stopped:
                st.warning("🛑 Scan stopped by user.")
                break

            if st.session_state.scan_paused:
                st.info("⏸️ Paused... Click again to resume.")
                st.session_state.current_frame_num = frame_num
                st.stop()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_display.markdown(f"🧮 Scanning frame: {frame_num}/{total_frames} ({scan_option})")

            faces = app.get(frame)
            new_match_added = False

            for face in faces:
                box = face.bbox.astype(int)
                face_crop = frame[box[1]:box[3], box[0]:box[2]]
                if face_crop.size == 0 or is_blurry(face_crop):
                    continue

                face_emb_norm = face.embedding / np.linalg.norm(face.embedding)

                for query_name, query_emb in query_embeddings:
                    sim = np.dot(query_emb, face_emb_norm)
                    if sim > similarity_threshold:
                        timestamp = str(datetime.timedelta(seconds=frame_num / fps))
                        filename = os.path.join(output_dir, f"match_{query_name}_at_{timestamp.replace(':','-')}_sim_{sim:.2f}.jpg")
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"sim: {sim:.2f}", (box[0], box[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imwrite(filename, frame)

                        st.session_state.all_matches.append((video_file.name, frame_num, timestamp, query_name, sim, filename))
                        st.session_state.match_images.insert(0, (filename, f"{query_name} in {video_file.name} at {timestamp} (Sim: {sim:.2f})"))
                        new_match_added = True

            if new_match_added:
                df_live = pd.DataFrame(
                    reversed(st.session_state.all_matches),
                    columns=["Video", "Frame", "Timestamp", "Query Image", "Similarity", "Image Path"]
                )
                df_placeholder.markdown("### 📊 Live Match Table (Latest First)")
                df_placeholder.dataframe(df_live.drop(columns=["Image Path"]), use_container_width=True)

            processed_frames += 1
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            remaining = int((elapsed / processed_frames) * (total_frames_to_process - processed_frames)) if processed_frames > 0 else 0
            remaining_time_str = f"{remaining // 60:02.0f}:{remaining % 60:02.0f}"
            progress.progress(min(frame_num / total_frames, 1.0), text=f"Processing... ETA: {remaining_time_str} remaining")

            frame_num += frame_skip

        st.session_state.current_frame_num = 0
        cap.release()

    st.session_state.scan_complete = True

# Final display if scan completed or manually stopped
if st.session_state.scan_complete or st.session_state.scan_stopped:
    if st.session_state.all_matches:
        df_final = pd.DataFrame(
            st.session_state.all_matches,
            columns=["Video", "Frame", "Timestamp", "Query Image", "Similarity", "Image Path"]
        )
        st.success(f"✅ Found {len(df_final)} total matches across all videos.")
        st.markdown("### 📊 All Match Results")
        st.dataframe(df_final.drop(columns=["Image Path"]), use_container_width=True)

        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download Match Report (CSV)", data=csv, file_name="face_match_results.csv", mime="text/csv")

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for img_path, _ in st.session_state.match_images:
                zip_file.write(img_path, arcname=os.path.basename(img_path))
        zip_buffer.seek(0)
        st.download_button("🖼️ Download All Matched Images (ZIP)", data=zip_buffer, file_name="matched_faces.zip", mime="application/zip")

        st.markdown("### 🖼️ All Matched Frames")
        for img_path, caption in st.session_state.match_images:
            st.image(img_path, caption=caption, use_container_width=True)
    else:
        st.warning("No matches found across all videos.")
