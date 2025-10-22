# 🎥 Multimodal Video Retrieval System (AIC2025)

An efficient **multimodal video retrieval pipeline** designed to index and search through massive video datasets using **CLIP embeddings**, **FAISS**, and metadata extracted from **OCR**, **ASR**, **Object Detection**, and **Scene Recognition**.

This project is built as part of **AI Challenge 2025**.  
It focuses on creating a scalable yet accurate retrieval system for large-scale video keyframes.

---

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

📧 Contact: [alexngo4work@example.com](mailto:alexngo4work@gmail.com)

```
