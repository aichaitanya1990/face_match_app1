import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
from insightface.app import FaceAnalysis

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

# Process if files uploaded
if query_img_files and video_files:
    # Extract embeddings from all query images
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

    all_matches = []

    # Process each video
    for video_file in video_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        output_dir = tempfile.mkdtemp()

        st.markdown(f"### 🔍 Processing video: {video_file.name}")
        progress = st.progress(0)

        def is_blurry(image, threshold=80):
            return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break            

            if frame_num % 30 != 0:
                frame_num += 1
                continue

            print(f"Processing frame {frame_num}/{total_frames} in video {video_file.name}")

            faces = app.get(frame)
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
                        filename = os.path.join(output_dir, f"match_{query_name}_{frame_num}_sim_{sim:.2f}.jpg")
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"sim: {sim:.2f}", (box[0], box[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imwrite(filename, frame)
                        all_matches.append((video_file.name, frame_num, timestamp, query_name, sim, filename))

            frame_num += 1
            progress.progress(min(frame_num / total_frames, 1.0))

        cap.release()

    # Display results
    if all_matches:
        df = pd.DataFrame(all_matches, columns=["Video", "Frame", "Timestamp", "Query Image", "Similarity", "Image Path"])
        st.success(f"✅ Found {len(df)} total matches across all videos.")
        st.dataframe(df.drop(columns=["Image Path"]))

        st.markdown("### 🖼️ All Matched Frames")
        for _, row in df.iterrows():
            st.image(row["Image Path"], caption=f"{row['Query Image']} in {row['Video']} at {row['Timestamp']} (Sim: {row['Similarity']:.2f})")
    else:
        st.warning("No matches found across all videos.")
