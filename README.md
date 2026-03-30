<div align="center">
   
# 🎥 Multimodal Video Retrieval System (AIC2025)

![last commit](https://img.shields.io/badge/last_commit-today-1abc9c?style=for-the-badge)
<img src="https://img.shields.io/badge/Python-100%25-blue?style=for-the-badge&logo=python&logoColor=white" alt="Language Stats" />
![languages count](https://img.shields.io/badge/languages-3-9b59b6?style=for-the-badge)
  
<p>Built with the tools and technologies:</p>
  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/OpenAI_CLIP-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI" />
<img src="https://img.shields.io/badge/Meta_FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Meta" />
<img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" alt="Google Colab" />
<img src="https://img.shields.io/badge/JSON-5E5C5C?style=for-the-badge&logo=json&logoColor=white" alt="JSON" />
<img src="https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white" alt="Markdown" />

![Build](https://img.shields.io/badge/CI-not_configured-lightgrey?style=flat-square)
![Coverage](https://img.shields.io/badge/coverage-not_measured-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
</div>

## 🧠 Overview

### 🔍 What It Does
Given a massive dataset of **video keyframes**, the system:
1. Extracts multimodal metadata:
   - Text (from **OCR**)
   - Speech (from **ASR**)
   - Objects (via **YOLOv7**)
   - Scene context (via **Places365** or similar)
2. Generates 512-D **CLIP visual embeddings**
3. Builds a **FAISS vector index** (L2 / HNSW / GPU)
4. Retrieves the most relevant frames for a **query image or text**

---

## 🏗️ Pipeline Architecture

```text
             ┌────────────────────────────┐
             │         Dataset            │
             │   (Video Keyframes)        │
             └────────────┬───────────────┘
                          │
         ┌────────────────┴────────────────┐
         │   Multimodal Feature Extractor  │
         │  (OCR + ASR + YOLO + Scene)    │
         └────────────────┬────────────────┘
                          │
                 ┌────────┴────────┐
                 │ CLIP Embeddings │  → (512d normalized vectors)
                 └────────┬────────┘
                          │
               ┌──────────┴──────────┐
               │  FAISS Index Build  │  ← FlatL2 / HNSW / GPU
               └──────────┬──────────┘
                          │
                    🔎 Retrieval Engine
                          │
             ┌────────────┴─────────────┐
             │  Query (Image / Text)    │
             └────────────┬─────────────┘
                          ▼
                 ✅ Ranked Results
````

---

## 🧩 Components

| Module                 | Description                            |
| ---------------------- | -------------------------------------- |
| `embedding_model_2.py` | Generate CLIP embeddings for keyframes |
| `object_detector.py`   | Detect visual objects using YOLOv7     |
| `ocr_extractor.py`     | Extract word-level text via EasyOCR    |
| `build_faiss_index.py` | Build **FAISS FlatL2** index           |
| `search_text_frame.py` | Perform similarity search via FAISS    |
| `data/embeddings_v2/`  | Stored CLIP embeddings and metadata    |

---

## 📁 Directory Structure

```
backend/
├─ app/
│  ├─ models/
│  │  ├─ embedding_model_2.py
│  │  ├─ object_detector.py
│  │  └─ build_faiss_index.py
│  └─ utils/
│     └─ ocr_extractor.py
├─ data/
│  ├─ embeddings_v2/
│  │  ├─ embeds/frame_clip/<video_id>/<frame_idx>.npy
│  │  └─ meta/frame_meta/<video_id>/<frame_idx>.json
│  └─ faiss_index_v2/
└─ requirements.txt
```

---

## ⚙️ Setup

### Option 1. Local (CPU)

```bash
git clone https://github.com/<your_username>/<repo_name>.git
cd <repo_name>
pip install -r requirements.txt
```

### Option 2. Google Colab (with GPU)

```bash
!pip install --no-cache-dir faiss-gpu-cu11==1.7.4.post2
!pip install ftfy regex tqdm pillow torch torchvision
```

---

## 🚀 Build CLIP Embeddings

```bash
python embedding_model_2.py --input-dir data/keyframes --output-dir data/embeddings_v2
```

---

## 🧱 Build FAISS Index

### FlatL2 (maximum precision)

```bash
python build_faiss_index.py
```

### HNSW (fast, tunable)

```bash
python build_hnsw_index.py --use-gpu --hnsw-m 32 --ef-construction 200 --ef-search 500
```

> 📌 Notes:
>
> * `ef-construction`: Quality during index build (↑ accuracy, ↓ speed)
> * `ef-search`: Accuracy at query time (↑ = more accurate)

---

## 🔍 Query Example

```python
import faiss, numpy as np, json

index = faiss.read_index("data/faiss_index_v2/frame_clip_index.faiss")
id_map = json.load(open("data/faiss_index_v2/frame_clip_id_mapping.json"))
query_vec = np.load("query_clip.npy").astype(np.float32).reshape(1, -1)

k = 5
D, I = index.search(query_vec, k)
for dist, idx in zip(D[0], I[0]):
    print(f"Match UID: {id_map[str(idx)]}, Distance: {dist}")
```

---

## 🧪 Evaluation

| Metric                       | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| Recall@K                     | Proportion of relevant keyframes found within top-K |
| Precision                    | Correct results / total retrieved                   |
| Mean Average Precision (mAP) | Overall ranking quality                             |

---

## 💡 Future Work

* Fuse multimodal embeddings (visual + textual)
* Improve OCR/ASR noise robustness
* Implement re-ranking with cross-modal similarity
* Integrate UI-based visual search interface

---

## 👨‍💻 Tech Stack

| Component         | Technology                  |
| ----------------- | --------------------------- |
| Embedding         | OpenAI CLIP (ViT-B/32)      |
| Indexing          | FAISS (FlatL2 / HNSW)       |
| Object Detection  | YOLOv7                      |
| OCR               | EasyOCR                     |
| ASR               | Whisper                     |
| Scene Recognition | Places365                   |
| Backend           | Python 3.11                 |
| GPU Support       | FAISS GPU (CUDA 11/12)      |
| Deployment        | Google Colab / Local Server |

---

## 📜 License

This project is open-source under the **MIT License**.

---

## 🤝 Acknowledgements

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [FAISS by Meta](https://github.com/facebookresearch/faiss)
* [YOLOv7 by WongKinYiu](https://github.com/WongKinYiu/yolov7)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [Whisper by OpenAI](https://github.com/openai/whisper)

---

## 🏆 Author

**Data Vision Team**

💼 AI Challenge 2025

📧 Contact: [alexngo4work@gmail.com](mailto:alexngo4work@gmail.com)

```
