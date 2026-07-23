import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer

# 1. Page layout and header initialization
st.set_page_config(page_title="Universal Media Organizer", layout="wide")
st.title("📂 Free Multi-Source Unsupervised Media Organizer")
st.write("Test with built-in cloud samples, upload external local assets, or combine both sources seamlessly.")

# 2. Setup and cache the AI architecture 
@st.cache_resource
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_clip_model()

# 3. Helper Functions for Processing Visual Frameworks
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

def fetch_sample_bytes(sample_name):
    """Generates a clean synthetic image in-memory to prevent web link dropouts completely."""
    if "cat" in sample_name.lower():
        color = [210, 105, 30]   # Chocolate Brown tint
    elif "car" in sample_name.lower():
        color = [220, 20, 60]    # Crimson Red tint
    elif "nature" in sample_name.lower():
        color = [34, 139, 34]    # Forest Green tint
    else:
        color = [128, 128, 128]  # Slate Grey tint
        
    img_array = np.full((300, 300, 3), color, dtype=np.uint8)
    image = Image.fromarray(img_array)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()

# 4. Interactive Sidebar Configuration Panel
st.sidebar.header("🕹️ Control Dashboard")
raw_concepts = st.sidebar.text_input("Target Grouping Keywords:", "cat, dog, car, nature")
concepts = [c.strip().lower() for c in raw_concepts.split(",") if c.strip()]
confidence_threshold = st.sidebar.slider("AI Confidence Cutoff Threshold", 0.0, 1.0, 0.18)

st.sidebar.write("---")
st.sidebar.subheader("📦 Data Source Options")

# Option #1: The Built-in Testing Samples toggle checkbox
use_samples = st.sidebar.checkbox("Load Built-In Cloud Samples", value=True, 
                                  help="Enables quick system testing with generated images if you don't have local files ready.")

# Fixed internal sample system
SAMPLE_URLS = [
    {"name": "internal_sample_cat.jpg", "url": "local"},
    {"name": "internal_sample_car.jpg", "url": "local"},
    {"name": "internal_sample_nature.jpg", "url": "local"}
]

# Primary list to hold combined inputs
aggregated_media_queue = []

# Process Sample Files if checked
if use_samples:
    st.sidebar.caption("🟢 Live Cloud Samples active.")
    for sample in SAMPLE_URLS:
        aggregated_media_queue.append({
            "name": sample["name"],
            "data_source": "url",
            "url_link": sample["url"]
        })

# Option #2: External Upload drag-and-drop module
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
            "data_source": "bytes",
            "raw_bytes": f.read()
        })

# ========================================================
# CORE MULTIMODAL ALL-IN-ONE PIPELINE
# ========================================================
if aggregated_media_queue and concepts:
    st.write("---")
    st.subheader("⚙️ Unified AI Processing Lane")
    
    # Calculate target anchor embeddings for keyword alignment
    concept_embeddings = model.encode([f"a photo of a {c}" for c in concepts])
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    
    # Initialize dictionary categorization buckets
    output_buckets = {c: [] for c in concepts}
    output_buckets["unclassified"] = []  # The Ungrouped bucket
    
    for asset in aggregated_media_queue:
        name = asset["name"]
        _, file_extension = os.path.splitext(name.lower())
        if not file_extension:
            file_extension = ".jpg"
            
        target_bytes = None
        
        # Route file content extraction paths based on source location
        if asset["data_source"] == "bytes":
            target_bytes = asset["raw_bytes"]
        elif asset["data_source"] == "url":
            with st.spinner(f"Generating memory asset matrix: {name}..."):
                target_bytes = fetch_sample_bytes(asset["name"])
                
        if target_bytes is None:
            st.error(f"Skipped asset line item processing for: {name}")
            continue
            
        parsed_visual_matrix = None
        
        # Decode data stream blocks into raw AI visual frames
        if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
            try:
                parsed_visual_matrix = Image.open(BytesIO(target_bytes)).convert("RGB")
            except:
                pass
        elif file_extension in ['.mp4', '.avi', '.mov']:
            parsed_visual_matrix = extract_video_frame(target_bytes, file_extension)
            
        if parsed_visual_matrix is not None:
            # Execute feature array mapping extraction
            extracted_vector = model.encode(parsed_visual_matrix)
            extracted_vector = extracted_vector / np.linalg.norm(extracted_vector)
            
            # Match matrix coordinates against target concept anchors
            match_scores = np.dot(concept_embeddings, extracted_vector)
            top_match_idx = np.argmax(match_scores)
            max_confidence_score = match_scores[top_match_idx]
            
            # Categorize the item based on the threshold setting
            if max_confidence_score >= confidence_threshold:
                assigned_category = concepts[top_match_idx]
            else:
                assigned_category = "unclassified"
                
            origin_type = "Built-In Sample" if asset["data_source"] == "url" else "External Upload"
            
            output_buckets[assigned_category].append({
                "name": name,
                "frame": parsed_visual_matrix,
                "score": max_confidence_score,
                "origin": origin_type
            })
            st.success(f"⚡ Mapped **{name}** ({origin_type.upper()}) -> **[{assigned_category.upper()}]**")
        else:
            st.error(f"⚠️ Formatting error parsing input streams for file: {name}")

    # ========================================================
    # WEB UI RENDERING GRID (FIXED FOR UNGROUPED ITEMS)
    # ========================================================
    st.write("---")
    st.subheader("📂 Dynamic Virtual Output Folders")
    
    # Loop over all buckets inside output_buckets rather than just user concepts
    for group_title, contents_list in output_buckets.items():
        if contents_list:
            # Highlight unclassified items cleanly for user clarity
            display_title = f"⚠️ {group_title.upper()} / UNGROUPED" if group_title == "unclassified" else f"📁 {group_title.upper()}"
            
            with st.expander(f"{display_title} ({len(contents_list)} items grouped)", expanded=True):
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
