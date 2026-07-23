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
    """Reads samples.config and programmatically builds texturally distinct images to calibrate the raw CLIP vision model."""
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
                    
                    # Force overwrite the old flat blocks to generate rich texture patterns
                    img_matrix = np.zeros((400, 400, 3), dtype=np.uint8)
                    
                    # Create basic mathematical frequency noise to simulate organic textures
                    x = np.linspace(0, 10, 400)
                    y = np.linspace(0, 10, 400)
                    X, Y = np.meshgrid(x, y)
                    noise = (np.sin(X) * np.cos(Y) + 1) / 2 * 30  # Subtle textured variance array
                    
                    if "cat" in generator_type:
                        # 1. Warm Tabby Cat Ginger Textures
                        img_matrix[:, :, 0] = np.clip(30 + noise, 0, 255)   # Blue channel
                        img_matrix[:, :, 1] = np.clip(120 + noise * 1.5, 0, 255) # Green channel
                        img_matrix[:, :, 2] = np.clip(230 + noise, 0, 255)  # Red channel
                        # Draw cat eye silhouettes to anchor cat face geometry features
                        cv2.circle(img_matrix, (150, 180), 15, (20, 230, 230), -1)
                        cv2.circle(img_matrix, (250, 180), 15, (20, 230, 230), -1)
                    elif "car" in generator_type:
                        # 2. Metallic Racing Car Geometries
                        img_matrix[:, :, 2] = np.clip(200 + noise, 0, 255) # Solid Red Chassis base
                        # Draw structural wheel shapes and windshield lines to signal vehicle design layouts
                        cv2.rectangle(img_matrix, (40, 220), (360, 320), (30, 30, 30), -1)
                        cv2.circle(img_matrix, (100, 320), 40, (10, 10, 10), -1)
                        cv2.circle(img_matrix, (300, 320), 40, (10, 10, 10), -1)
                    else:
                        # 3. Fluffy Canine Shaggy Fur Textures
                        img_matrix[:, :, 0] = np.clip(20 + noise, 0, 255)   # Deep Earthy Brown base
                        img_matrix[:, :, 1] = np.clip(60 + noise, 0, 255)
                        img_matrix[:, :, 2] = np.clip(110 + noise, 0, 255)
                        # Draw floppy ear and snout coordinates to map canine features
                        cv2.ellipse(img_matrix, (200, 240), (40, 25), 0, 0, 360, (15, 15, 15), -1)
                        
                    cv2.imwrite(full_file_path, img_matrix)
                    
                    if os.path.exists(full_file_path):
                        sample_manifest.append({"name": filename, "path": full_file_path})
                except ValueError:
                    st.sidebar.warning(f"⚠️ Malformed config line {idx+1}")
                    
    return sample_manifest

# Automatically parse config and trigger setup sync on app initialization
SAMPLE_MANIFEST = load_and_sync_samples()

# 3. Model Caching and Video Processor
@st.cache_resource
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_clip_model()

def extract_video_frame(file_bytes, ext):
    """Slices open a video data stream and returns a combined, averaged image from multiple points."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define three distinct time points to capture action segments
    frame_checkpoints = [total_frames // 4, total_frames // 2, (3 * total_frames) // 4]
    captured_frames = []
    
    for pos in frame_checkpoints:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        success, frame = cap.read()
        if success:
            # Convert OpenCV BGR array to standard PIL RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_frames.append(rgb_frame)
            
    cap.release()
    os.unlink(temp_path)
    
    if len(captured_frames) > 0:
        # Take the mathematical pixel mean across all captured frames to create a clean visual summary
        averaged_matrix = np.mean(captured_frames, axis=0).astype(np.uint8)
        return Image.fromarray(averaged_matrix)
        
    return None
    
# 4. Sidebar Controls and Media Aggregation
st.sidebar.header("🕹️ Control Dashboard")
raw_concepts = st.sidebar.text_input("Target Grouping Keywords:", "cat, dog, car, nature")
concepts = [c.strip().lower() for c in raw_concepts.split(",") if c.strip()]
confidence_threshold = st.sidebar.slider("AI Confidence Cutoff Threshold", 0.0, 1.0, 0.26)

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

# 5. Core Multimodal Processing AI Loop (With Multi-Prompt Realignment Fixes)
if aggregated_media_queue and concepts:
    st.write("---")
    st.subheader("⚙️ Unified AI Processing Lane")
    
    # Generate multi-context description matrices for each target keyword
    concept_list_embeddings = []
    for c in concepts:
        prompt_templates = [
            f"a photo of a {c}",
            f"a close up photograph of a {c}",
            f"a domestic pet {c}",
            f"a beautiful clean {c}",
            f"the silhouette of a clear {c}"
        ]
        template_vectors = model.encode(prompt_templates)
        averaged_concept_vector = np.mean(template_vectors, axis=0)
        concept_list_embeddings.append(averaged_concept_vector)
        
    concept_embeddings = np.array(concept_list_embeddings)
    concept_embeddings = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
    
    output_buckets = {c: [] for c in concepts}
    output_buckets["unclassified"] = []
    
    # Keep a flat list of all items with their normalized vectors for the search engine feature
    all_processed_items = []
    
    for asset in aggregated_media_queue:
        name = asset["name"]
        _, file_extension = os.path.splitext(name.lower())
        if not file_extension:
            file_extension = ".jpg"
            
        parsed_visual_matrix = None
        extracted_vector = None
        target_bytes = None
        origin_type = "External Upload" if asset["data_source"] == "user_bytes" else "Built-In Sample"
        
        if asset["data_source"] == "server_disk":
            with open(asset["file_path"], "rb") as disk_file:
                target_bytes = disk_file.read()
            parsed_visual_matrix = Image.open(BytesIO(target_bytes)).convert("RGB")
            extracted_vector = model.encode(parsed_visual_matrix)
        elif asset["data_source"] == "user_bytes":
            target_bytes = asset["raw_bytes"]
            if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
                try:
                    parsed_visual_matrix = Image.open(BytesIO(target_bytes)).convert("RGB")
                except:
                    pass
            elif file_extension in ['.mp4', '.avi', '.mov']:
                parsed_visual_matrix = extract_video_frame(target_bytes, file_extension)
            if parsed_visual_matrix is not None:
                extracted_vector = model.encode(parsed_visual_matrix)

        if extracted_vector is not None and parsed_visual_matrix is not None and target_bytes is not None:
            # Normalize vector weights to unit scale
            normalized_vector = extracted_vector / np.linalg.norm(extracted_vector)
            match_scores = np.dot(concept_embeddings, normalized_vector)
            top_match_idx = np.argmax(match_scores)
            max_confidence_score = match_scores[top_match_idx]
            
            if max_confidence_score >= confidence_threshold:
                assigned_category = concepts[top_match_idx]
            else:
                assigned_category = "unclassified"
                
            item_payload = {
                "name": name, 
                "frame": parsed_visual_matrix, 
                "score": max_confidence_score, 
                "origin": origin_type,
                "raw_file_bytes": target_bytes,
                "vector": normalized_vector
            }
            
            output_buckets[assigned_category].append(item_payload)
            all_processed_items.append(item_payload)
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
                        
                        archive_path = f"{group_title}/{grid_item['name']}"
                        zip_file.writestr(archive_path, grid_item["raw_file_bytes"])

    # ========================================================
    # NEW: INTERACTIVE SEMANTIC SEARCH BOX
    # ========================================================
    st.write("---")
    st.subheader("🔍 Conceptual Media Search Engine")
    search_query = st.text_input("Type an unstructured sentence to query your media pool instantly:", placeholder="e.g., a pet playing outside or a fast vehicle")
    
    if search_query:
        # Encode the search query sentence into vector space coordinates
        query_vector = model.encode(search_query)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Calculate similarity scores for all processed files
        search_results = []
        for item in all_processed_items:
            similarity = np.dot(item["vector"], query_vector)
            search_results.append((similarity, item))
            
        # Sort results from highest match to lowest match
        search_results.sort(key=lambda x: x[0], reverse=True)
        
        st.write(f"Showing best matching files for: *\"{search_query}\"*")
        search_cols = st.columns(4)
        for idx, (sim_score, item) in enumerate(search_results[:4]): # Render top 4 best matches
            with search_cols[idx % 4]:
                st.image(item["frame"], use_container_width=True)
                st.caption(f"**{item['name']}**\n\nSemantic Relevance: {sim_score:.2f}")

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
