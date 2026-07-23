# 📂 AI Multi-Source Unsupervised Media Organizer

Developed with ❤️ by **Srinivasta**

An advanced, free, and completely web-based unsupervised machine learning application built with **Streamlit**, **OpenAI's CLIP model**, and **Plotly**. This application automatically interprets, aligns, and categorizes unstructured text, images, and video frames by their true conceptual meaning, completely bypassing the traditional multi-modal formatting gap.

---

## 🚀 Key Features

* **Zero Cloud Costs**: Powered entirely by open-source models (`clip-ViT-B-32`) running natively inside memory containers—no paid API keys, credits, or registration required.
* **Dual-Input Engine**: Test out-of-the-box using background-downloaded cloud sample paths or drag-and-drop your own personal `.png`, `.jpg`, or `.mp4` files seamlessly.
* **Dynamic Content Search**: A built-in cross-modal semantic search prompt allows users to write natural sentences to query and filter the active media pool on the fly.
* **Smart Cutoff Sliders**: Adjust confidence thresholds dynamically to let real photographic pixels pass into target folders while isolating low-contrast dummy blocks into an Ungrouped drawer.
* **One-Click Local Exports**: Packages organized virtual outputs into physical subdirectory tree archives (`/cat`, `/dog`, `/car`) inside a single structured `.zip` download wrapper.
* **Visual Data Analytics**: Integrates responsive horizontal Plotly charts providing clear mathematical proof behind every algorithmic matching decision.

---

## 🛠️ Repository File Architecture

```text
multimodal-media-sorter/
├── app.py                # Core Python application managing the AI & Streamlit layers
├── requirements.txt      # Automated Linux server deployment configurations
├── samples.config        # Decoupled text catalog handling sample filenames and URLs
└── README.md             # Project documentation page guidelines
```

---

## ⚙️ Configuration Setup

### 1. Requirements File (`requirements.txt`)
Ensure your dependencies match this list to ensure safe deployment inside headless server spaces:
```text
streamlit
sentence-transformers
pillow
opencv-python-headless
numpy
scikit-learn
requests
plotly
pandas
```

### 2. Samples Configuration (`samples.config`)
Populate your testing file using a comma-separated filename and direct URL layout:
```text
test_cat.jpg,generate_cat_pattern
test_car.jpg,generate_car_pattern
test_dog.jpg,generate_dog_pattern
```

---

## 📦 How to Launch Live on the Web

1. Push all files to your public repository on your personal **GitHub** profile.
2. Navigate to [Streamlit Community Cloud](https://multimodal-media-sorter-zg3j83appudnqpfhyhutucc.streamlit.app/) and log in using your GitHub credentials.
3. Click the **Create app** dashboard button.
4. Select this repository branch track, specify the main execution path to `app.py`, and click **Deploy!**
5. Streamlit will configure the container, install requirements, download the neural weights, and host your app live on the internet forever at zero cost.

---
*Created by Srinivasta — Powered by Open-Source Artificial Intelligence.*
