import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

# Page layout and header initialization
st.set_page_config(page_title="Universal Media Organizer", layout="wide")
st.title("📂 Free Multi-Source Unsupervised Media Organizer")
st.write("Test with built-in downloaded cloud samples, upload external local assets, or combine both sources seamlessly.")

def load_and_sync_samples():
    """Reads external samples.config file and downloads missing assets onto the server disk."""
    local_target_directory = "raw_unorganized_files"
    os.makedirs(local_target_directory, exist_ok=True)
    
    config_file = "samples.config"
    sample_manifest = []
    
    # Check if the configuration file exists
    if not os.path.exists(config_file):
        st.sidebar.error(f"❌ File Not Found: '{config_file}' is missing from the server root directory.")
        return []
        
    st.sidebar.success(f"📂 Found '{config_file}' file on server!")
    
    # Read lines and download assets dynamically
    with open(config_file, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    filename, url = line.split(",", 1)
                    filename = filename.strip()
                    url = url.strip()
                    
                    full_file_path = os.path.join(local_target_directory, filename)
                    
                    # Try downloading the image file
                    if not os.path.exists(full_file_path):
                        try:
                            # FIX: Add User-Agent headers to completely bypass the 403 Forbidden blocking
                            browser_headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                            }
                            response = requests.get(url, headers=browser_headers, timeout=15)
                            
                            if response.status_code == 200:
                                with open(full_file_path, "wb") as file_handler:
                                    file_handler.write(response.content)
                            else:
                                st.sidebar.error(f"❌ Web Error on line {idx+1}: Received code {response.status_code} for {filename}")
                        except Exception as download_error:
                            st.sidebar.error(f"❌ Connection Error on line {idx+1}: {download_error}")
                            
                    if os.path.exists(full_file_path):
                        sample_manifest.append({"name": filename, "path": full_file_path})
                except ValueError:
                    st.sidebar.warning(f"⚠️ Malformed config line {idx+1}: '{line}'")
                    
    return sample_manifest

# Automatically parse config and trigger setup sync on app initialization
SAMPLE_MANIFEST = load_and_sync_samples()

# Setup and cache the AI architecture 
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
# Interactive Sidebar Configuration Panel
st.sidebar.header("🕹️ Control Dashboard")
raw_concepts = st.sidebar.text_input("Target Grouping Keywords:", "cat, dog, car, nature")
concepts = [c.strip().lower() for c in raw_concepts.split(",") if c.strip()]
confidence_threshold = st.sidebar.slider("AI Confidence Cutoff Threshold", 0.0, 1.0, 0.18)

st.sidebar.write("---")
st.sidebar.subheader("📦 Data Source Options")

# Checkbox option to include the background-downloaded test samples
use_samples = st.sidebar.checkbox("Load Built-In Cloud Samples", value=True, 
                                  help="Enables quick system testing with dynamically synced repository files.")

# Primary processing queue list
aggregated_media_queue = []

# Method 1: Feed files dynamically from server storage if synced manifest exists
if use_samples and SAMPLE_MANIFEST:
    st.sidebar.caption(f"🟢 Synchronized {len(SAMPLE_MANIFEST)} Cloud Samples active.")
    for sample in SAMPLE_MANIFEST:
        aggregated_media_queue.append({
            "name": sample["name"],
            "data_source": "server_disk",
            "file_path": sample["path"]
        })

# Method 2: Feed external uploaded files from drag-and-drop panel
st.subheader("📤 External Uploads Panel")
external_files = st.file_uploader(
    "Drag and drop files from your device to combine them with the cloud samples:", 
    type=["png", "jpg", "jpeg", "webp", "mp4", "avi", "mov"], 
    accept_multiple_files=True
)

if external_files:
    for f in external_files:
        aggregated_media_queue.append({
            "name": f.name,
            "data_source": "user_bytes",
            "raw_bytes": f.read()
        })
if aggregated_media_queue and concepts:
    st.write("---")
    st.subheader("⚙️ Unified AI Processing Lane")
    
    # Calculate target anchor embeddings for keyword alignment
    concept_embeddings = model.encode([f"a photo of a {c}" for c in concepts])
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    
    # Initialize categorization buckets
    output_buckets = {c: [] for c in concepts}
    output_buckets["unclassified"] = []  # Explicitly preserve the ungrouped bucket
    
    for asset in aggregated_media_queue:
        name = asset["name"]
        _, file_extension = os.path.splitext(name.lower())
        if not file_extension:
            file_extension = ".jpg"
            
        target_bytes = None
        
        # Pull raw file bytes depending on where the item originated
        if asset["data_source"] == "user_bytes":
            target_bytes = asset["raw_bytes"]
            origin_type = "External Upload"
        elif asset["data_source"] == "server_disk":
            with open(asset["file_path"], "rb") as disk_file:
                target_bytes = disk_file.read()
            origin_type = "Built-In Sample"
                
        if target_bytes is None:
            st.error(f"Skipped processing for: {name}")
            continue
            
        parsed_visual_matrix = None
        
        # Decode binary stream into an AI-ready visual matrix
        if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
            try:
                parsed_visual_matrix = Image.open(BytesIO(target_bytes)).convert("RGB")
            except:
                pass
        elif file_extension in ['.mp4', '.avi', '.mov']:
            parsed_visual_matrix = extract_video_frame(target_bytes, file_extension)
            
        if parsed_visual_matrix is not None:
            # Extract features and normalize vector weights
            extracted_vector = model.encode(parsed_visual_matrix)
            extracted_vector = extracted_vector / np.linalg.norm(extracted_vector)
            
            # Run comparison math against user tags
            match_scores = np.dot(concept_embeddings, extracted_vector)
            top_match_idx = np.argmax(match_scores)
            max_confidence_score = match_scores[top_match_idx]
            
            # Map item to a keyword folder or send it to the ungrouped folder
            if max_confidence_score >= confidence_threshold:
                assigned_category = concepts[top_match_idx]
            else:
                assigned_category = "unclassified"
                
            output_buckets[assigned_category].append({
                "name": name,
                "frame": parsed_visual_matrix,
                "score": max_confidence_score,
                "origin": origin_type
            })
            st.success(f"⚡ Mapped **{name}** ({origin_type.upper()}) -> **[{assigned_category.upper()}]**")
        else:
            st.error(f"⚠️ Formatting error parsing input stream for: {name}")
    # ========================================================
    # RENDER WEB UI GRID
    # ========================================================
    st.write("---")
    st.subheader("📂 Dynamic Virtual Output Folders")
    
    # Loop over all available buckets so 'unclassified' renders safely
    for group_title, contents_list in output_buckets.items():
        if contents_list:
            # Label folder gracefully if it contains the ungrouped lower-threshold items
            if group_title == "unclassified":
                folder_label = "⚠️ UNCLASSIFIED / UNGROUPED"
            else:
                folder_label = f"📁 {group_title.upper()}"
                
            with st.expander(f"{folder_label} ({len(contents_list)} items grouped)", expanded=True):
                grid_columns = st.columns(4)
                for index, grid_item in enumerate(contents_list):
                    with grid_columns[index % 4]:
                        st.image(grid_item["frame"], use_container_width=True)
                        st.caption(
                            f"**Name:** {grid_item['name']}\n\n"
                            f"**Source:** {grid_item['origin']}\n\n"
                            f"**Confidence:** {grid_item['score']:.2f}"
                        )
else:
    st.warning("All input queues empty. Toggle on 'Load Built-In Cloud Samples' or upload external files to start.")
