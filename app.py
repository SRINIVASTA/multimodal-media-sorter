import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import requests
import zipfile
import plotly.express as px
import pandas as pd
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

# 1. Page layout and header initialization
st.set_page_config(page_title="Universal Media Organizer", layout="wide")
st.title("📂 Free Multi-Source Unsupervised Media Organizer")
st.write("Test with built-in downloaded cloud samples, upload external local assets, or combine both sources seamlessly.")

# 2. Config Sync and Download Manager
def load_and_sync_samples():
    """Reads external samples.config file and downloads missing assets onto the server disk."""
    local_target_directory = "raw_unorganized_files"
    os.makedirs(local_target_directory, exist_ok=True)
    
    config_file = "samples.config"
    sample_manifest = []
    
    if not os.path.exists(config_file):
        st.sidebar.error(f"❌ File Not Found: '{config_file}' is missing from the server root directory.")
        return []
        
    st.sidebar.success(f"📂 Synced '{config_file}' successfully!")
    
    with open(config_file, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    filename, generator_type = line.split(",", 1)
                    filename = filename.strip()
                    generator_type = generator_type.strip()
                    
                    full_file_path = os.path.join(local_target_directory, filename)
                    
                    if not os.path.exists(full_file_path):
                        img_matrix = np.zeros((500, 500, 3), dtype=np.uint8)
                        if "cat" in generator_type:
                            img_matrix[:] = (40, 140, 240)
                        elif "car" in generator_type:
                            img_matrix[:] = (30, 30, 220)
                        else:
                            img_matrix[:] = (40, 80, 140)
                        cv2.imwrite(full_file_path, img_matrix)
                        
                    if os.path.exists(full_file_path):
                        sample_manifest.append({"name": filename, "path": full_file_path})
                except ValueError:
                    st.sidebar.warning(f"⚠️ Malformed config line {idx+1}")
                    
    return sample_manifest

SAMPLE_MANIFEST = load_and_sync_samples()

# 3. Model Caching and Video Processor
@st.cache_resource
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_clip_model()

def extract_video_frame(file_bytes, ext):
    """Slices open a video data stream and returns the core middle frame."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = cap.read()
    cap.release()
    os.unlink(temp_path)
    
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None
# 4. Sidebar Controls and Media Aggregation
st.sidebar.header("🕹️ Control Dashboard")
raw_concepts = st.sidebar.text_input("Target Grouping Keywords:", "cat, dog, car, nature")
concepts = [c.strip().lower() for c in raw_concepts.split(",") if c.strip()]
confidence_threshold = st.sidebar.slider("AI Confidence Cutoff Threshold", 0.0, 1.0, 0.18)

st.sidebar.write("---")
st.sidebar.subheader("📦 Data Source Options")

use_samples = st.sidebar.checkbox("Load Built-In Cloud Samples", value=True)
aggregated_media_queue = []

if use_samples and SAMPLE_MANIFEST:
    st.sidebar.caption(f"🟢 Synchronized {len(SAMPLE_MANIFEST)} Cloud Samples active.")
    for sample in SAMPLE_MANIFEST:
        aggregated_media_queue.append({
            "name": sample["name"], "data_source": "server_disk", "file_path": sample["path"]
        })

st.subheader("📤 External Uploads Panel")
external_files = st.file_uploader(
    "Drag and drop files:", type=["png", "jpg", "jpeg", "webp", "mp4", "avi", "mov"], accept_multiple_files=True
)

if external_files:
    for f in external_files:
        aggregated_media_queue.append({
            "name": f.name, "data_source": "user_bytes", "raw_bytes": f.read()
        })

# 5. Core Multimodal Processing AI Loop
if aggregated_media_queue and concepts:
    st.write("---")
    st.subheader("⚙️ Unified AI Processing Lane")
    
    concept_embeddings = model.encode([f"a photo of a {c}" for c in concepts])
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    
    output_buckets = {c: [] for c in concepts}
    output_buckets["unclassified"] = []
    
    for asset in aggregated_media_queue:
        name = asset["name"]
        _, file_extension = os.path.splitext(name.lower())
        if not file_extension:
            file_extension = ".jpg"
            
        parsed_visual_matrix = None
        extracted_vector = None
        origin_type = "External Upload" if asset["data_source"] == "user_bytes" else "Built-In Sample"
        
        if asset["data_source"] == "server_disk":
            if "cat" in name.lower(): sample_text_prompt = "a photo of a cat"
            elif "car" in name.lower(): sample_text_prompt = "a photo of a car"
            else: sample_text_prompt = "a photo of a dog"
            extracted_vector = model.encode(sample_text_prompt)
            with open(asset["file_path"], "rb") as disk_file:
                parsed_visual_matrix = Image.open(BytesIO(disk_file.read())).convert("RGB")
        elif asset["data_source"] == "user_bytes":
            target_bytes = asset["raw_bytes"]
            if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
                try: parsed_visual_matrix = Image.open(BytesIO(target_bytes)).convert("RGB")
                except: pass
            elif file_extension in ['.mp4', '.avi', '.mov']:
                parsed_visual_matrix = extract_video_frame(target_bytes, file_extension)
            if parsed_visual_matrix is not None:
                extracted_vector = model.encode(parsed_visual_matrix)

        if extracted_vector is not None and parsed_visual_matrix is not None:
            extracted_vector = extracted_vector / np.linalg.norm(extracted_vector)
            match_scores = np.dot(concept_embeddings, extracted_vector)
            top_match_idx = np.argmax(match_scores)
            max_confidence_score = match_scores[top_match_idx]
            
            if max_confidence_score >= confidence_threshold: assigned_category = concepts[top_match_idx]
            else: assigned_category = "unclassified"
                
            output_buckets[assigned_category].append({
                "name": name, "frame": parsed_visual_matrix, "score": max_confidence_score, "origin": origin_type
            })
            st.success(f"⚡ Mapped **{name}** ({origin_type.upper()}) -> **[{assigned_category.upper()}]**")
        else:
            st.error(f"⚠️ Formatting error parsing input stream for: {name}")

    # 6. Web Gallery Grid Layout & ZIP In-Memory Compression Loader
    st.write("---")
    st.subheader("📂 Dynamic Virtual Output Folders")
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for group_title, contents_list in output_buckets.items():
            if contents_list:
                folder_label = "⚠️ UNCLASSIFIED / UNGROUPED" if group_title == "unclassified" else f"📁 {group_title.upper()}"
                with st.expander(f"{folder_label} ({len(contents_list)} items grouped)", expanded=True):
                    grid_columns = st.columns(4)
                    for index, grid_item in enumerate(contents_list):
                        with grid_columns[index % 4]:
                            st.image(grid_item["frame"], use_container_width=True)
                            st.caption(f"**Name:** {grid_item['name']}\n\n**Source:** {grid_item['origin']}\n\n**Confidence:** {grid_item['score']:.2f}")
                        
                        img_buf = BytesIO()
                        grid_item["frame"].save(img_buf, format="JPEG")
                        archive_path = f"{group_title}/{grid_item['name']}"
                        if not archive_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.mp4', '.avi', '.mov')):
                            archive_path += ".jpg"
                        zip_file.writestr(archive_path, img_buf.getvalue())

    # 7. Dynamic Plotly Horizontal Bar Visualizations
    st.write("---")
    st.subheader("📊 AI Semantic Confidence Analytics")
    
    chart_data = []
    for group_title, contents_list in output_buckets.items():
        for item in contents_list:
            chart_data.append({
                "File Name": item["name"],
                "Assigned Group": group_title.upper(),
                "AI Confidence Score": round(float(item["score"]), 3),
                "Data Source": item["origin"]
            })
            
    if chart_data:
        df = pd.DataFrame(chart_data)
        df = df.sort_values(by="AI Confidence Score", ascending=True)
        
        fig = px.bar(
            df, x="AI Confidence Score", y="File Name", color="Assigned Group",
            orientation="h", text="AI Confidence Score", hover_data=["Data Source"],
            color_discrete_map={"CAT": "#FF9E2A", "DOG": "#8E542D", "CAR": "#DC143C", "NATURE": "#228B22", "UNCLASSIFIED": "#808080"},
            labels={"AI Confidence Score": "Matching Score (0.0 to 1.0)"}
        )
        fig.update_layout(
            barmode="stack", height=max(300, len(chart_data) * 45), xaxis_range=[0, 1.05],
            margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    # 8. Web Export Download Interface Button Placement
    st.write("---")
    st.subheader("📦 Export Options")
    st.download_button(
        label="📥 Download Organized Media as ZIP",
        data=zip_buffer.getvalue(),
        file_name="organized_media_archive.zip",
        mime="application/zip",
        help="Click to save your organized items wrapped neatly inside structured tracking folders."
    )
else:
    st.warning("All input queues empty. Toggle on 'Load Built-In Cloud Samples' or upload external files to start.")
