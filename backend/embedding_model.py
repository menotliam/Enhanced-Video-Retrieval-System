# Complete Video Embedding Pipeline
# Frame-level fusion + Object-level embeddings

import os, json, math, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Models
from transformers import AutoTokenizer, AutoModel, LayoutLMv3Processor, LayoutLMv3Model
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------ 1) Initialize Models ------------
def init_models():
    """Initialize all embedding models"""
    models = {}
    
    # PhoBERT for Vietnamese text (ASR, Scene)
    print("Loading PhoBERT...")
    phobert_name = "vinai/phobert-base"
    models['phobert_tokenizer'] = AutoTokenizer.from_pretrained(phobert_name)
    models['phobert_model'] = AutoModel.from_pretrained(phobert_name).to(device)
    models['phobert_model'].eval()
    
    # LayoutLMv3 for OCR
    print("Loading LayoutLMv3...")
    layout_name = "microsoft/layoutlmv3-base"
    # Use external OCR (we provide texts and boxes), so disable internal OCR
    models['layout_processor'] = LayoutLMv3Processor.from_pretrained(layout_name, apply_ocr=False)
    models['layout_model'] = LayoutLMv3Model.from_pretrained(layout_name).to(device)
    models['layout_model'].eval()
    
    # CLIP for objects and global images
    print("Loading CLIP...")
    clip_name = "ViT-B-32"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_name, pretrained='openai',
        device=device,
        force_quick_gelu=False
    )
    models['clip_model'] = clip_model.to(device)
    models['clip_preprocess'] = clip_preprocess
    models['clip_model'].eval()
    
    # Get CLIP text tokenizer
    models['clip_tokenizer'] = open_clip.get_tokenizer(clip_name)
    
    return models

# ------------ 2) Data Loading Functions ------------
def load_frame_index(csv_path: str) -> Dict:
    """Load frame mapping from CSV"""
    df = pd.read_csv(csv_path)
    frame_map = {}
    
    for _, row in df.iterrows():
        frame_name = f"{int(row['n']):03d}.jpg"
        frame_map[frame_name] = {
            "frame_idx": int(row['frame_idx']),
            "pts_time": float(row['pts_time']),
            "fps": int(row.get('fps', 25))
        }
    return frame_map

def load_asr_data(json_path: str) -> Dict:
    """Load ASR segments"""
    with open(json_path, 'r', encoding='utf-8') as f:
        asr_data = json.load(f)
    return asr_data

def load_scene_data(json_path: str) -> Dict:
    """Load scene recognition data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    scene_map = {}
    for item in scene_data:
        filename = item.get('filename', '')
        scene_map[filename] = {
            'label_vi': item.get('label_vi', ''),
            'score': float(item.get('score', 0.0))
        }
    return scene_map

def load_ocr_data(json_path: str) -> Dict:
    """Load OCR data for specific frame"""
    if not os.path.exists(json_path):
        return {"texts": [], "bboxes": [], "confidences": []}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    # Filter by confidence threshold
    texts, bboxes, confidences = [], [], []
    min_conf = 0.5
    
    for item in ocr_data.get('words', []):
        conf = float(item.get('confidence', 0))
        if conf >= min_conf:
            texts.append(item.get('text', ''))
            # Convert bbox to xyxy format [0-1000]
            bbox = item.get('bbox', [0, 0, 0, 0])
            bboxes.append(bbox)
            confidences.append(conf)
    
    return {"texts": texts, "bboxes": bboxes, "confidences": confidences}

def load_object_data(json_path: str) -> List[Dict]:
    """Load object detection data for specific frame"""
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        obj_data = json.load(f)
    
    objects = []
    for i, obj in enumerate(obj_data.get('objects', [])):
        objects.append({
            'obj_id': obj.get('id', f"obj_{i}"),
            'label_vi': obj.get('label_vi', ''),
            'confidence': float(obj.get('confidence', 0)),
            'bbox_pixel': obj.get('bbox', []),
            'crop_path': obj.get('crop_path', '')
        })
    return objects

# ------------ 3) ASR Alignment ------------
def asr_text_for_frame(asr_segments: List[Dict], pts_time: float, slack: float = 0.5) -> Tuple[str, float]:
    """Find ASR text for specific frame timestamp"""
    # First try exact overlap
    for segment in asr_segments:
        start_time = float(segment.get('start_time', 0))
        end_time = float(segment.get('end_time', 0))
        
        if start_time <= pts_time <= end_time:
            transcript = segment.get('transcript', '').strip()
            return transcript, 1.0 if transcript else 0.0
    
    # If no exact match, find nearest within slack
    best_segment = None
    best_distance = float('inf')
    
    for segment in asr_segments:
        start_time = float(segment.get('start_time', 0))
        end_time = float(segment.get('end_time', 0))
        center_time = (start_time + end_time) / 2.0
        
        distance = abs(center_time - pts_time)
        if distance < best_distance and distance <= slack:
            best_distance = distance
            best_segment = segment
    
    if best_segment:
        transcript = best_segment.get('transcript', '').strip()
        confidence = max(0.1, 1.0 - best_distance / slack)  # distance-based confidence
        return transcript, confidence if transcript else 0.0
    
    return "", 0.0

# ------------ 4) Embedding Functions ------------
def embed_vietnamese_text(text: str, models: Dict) -> np.ndarray:
    """Embed Vietnamese text using PhoBERT"""
    if not text.strip():
        return np.zeros(768, dtype=np.float32)
    
    tokenizer = models['phobert_tokenizer']
    model = models['phobert_model']
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
        features_np = embeddings.cpu().numpy().squeeze().astype(np.float32)
        
        # L2-normalize the embedding
        norm = np.linalg.norm(features_np)
        if norm > 1e-12:
            features_np = features_np / norm
        
        return features_np

def embed_ocr_layoutlmv3(image_path: str, texts: List[str], bboxes: List[List[int]], models: Dict) -> np.ndarray:
    """Embed OCR using LayoutLMv3 (layout-aware)"""
    if not texts or not os.path.exists(image_path):
        return np.zeros(768, dtype=np.float32)
    
    try:
        processor = models['layout_processor']
        model = models['layout_model']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs (limit tokens to avoid overflow)
        max_tokens = 200
        if len(texts) > max_tokens:
            # Keep highest confidence ones (assume sorted by confidence)
            texts = texts[:max_tokens]
            bboxes = bboxes[:max_tokens]
        
        # Process inputs
        encoding = processor(
            image, 
            text=texts,
            boxes=bboxes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoding)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().squeeze().astype(np.float32)
            
    except Exception as e:
        print(f"Error in LayoutLMv3 embedding: {e}")
        return np.zeros(768, dtype=np.float32)

def embed_clip_image(image_path: str, models: Dict) -> np.ndarray:
    """Embed image using CLIP"""
    if not os.path.exists(image_path):
        return np.zeros(512, dtype=np.float32)
    
    try:
        clip_model = models['clip_model']
        preprocess = models['clip_preprocess']
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Get embedding
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            return image_features.cpu().numpy().squeeze().astype(np.float32)
            
    except Exception as e:
        print(f"Error in CLIP image embedding: {e}")
        return np.zeros(512, dtype=np.float32)

def embed_clip_text(text: str, models: Dict) -> np.ndarray:
    """Embed text using CLIP"""
    if not text.strip():
        return np.zeros(512, dtype=np.float32)
    
    try:
        clip_model = models['clip_model']
        tokenizer = models['clip_tokenizer']
        
        # Tokenize
        text_tokens = tokenizer([text]).to(device)  # Use global device
        
        # Get embedding
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            features_np = text_features.cpu().numpy().squeeze().astype(np.float32)
            
            # L2-normalize the embedding
            norm = np.linalg.norm(features_np)
            if norm > 1e-12:
                features_np = features_np / norm
            
            return features_np
            
    except Exception as e:
        print(f"Error in CLIP text embedding: {e}")
        return np.zeros(512, dtype=np.float32)

def embed_obj_crop(crop_path: str, label_vi: str, models: Dict, bbox_area: float = 0.05) -> np.ndarray:
    """Embed object crop with optional text label"""
    img_emb = embed_clip_image(crop_path, models)
    
    if label_vi and label_vi.strip():
        txt_emb = embed_clip_text(label_vi, models)
        # Dynamic mixing based on crop size/quality
        mix_ratio = 0.7 if bbox_area > 0.05 else 0.5
        return mix_ratio * img_emb + (1 - mix_ratio) * txt_emb
    
    return img_emb

# ------------ 5) Normalization & Fusion ------------
def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vector"""
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return vector / norm

def compute_adaptive_weights(asr_conf: float, ocr_conf: float, scene_score: float, has_objects: bool) -> Dict[str, float]:
    """Compute adaptive fusion weights based on quality scores"""
    # Default weights for OCR + Image only setup
    base_weights = {"asr": 0.0, "ocr": 0.0, "scene": 0.0, "gclip": 1.0}
    
    # If ASR is enabled (asr_conf > 0), adjust weights
    if asr_conf > 0:
        base_weights["asr"] = 0.3
        base_weights["ocr"] = 0.4
        base_weights["gclip"] = 0.3
    
    # If scene is enabled (scene_score > 0), adjust weights  
    if scene_score > 0:
        base_weights["scene"] = 0.2
        base_weights["ocr"] = 0.4
        base_weights["gclip"] = 0.4
    
    # Adjust based on confidence for enabled modalities
    if asr_conf > 0 and asr_conf < 0.3:  # Low ASR quality
        base_weights["asr"] *= 0.5
        base_weights["ocr"] += 0.1
    
    # if ocr_conf < 0.3:  # Low OCR quality
    #     base_weights["ocr"] *= 0.5
    #     base_weights["gclip"] += 0.1
    
    # Adjust for object-focused queries
    if has_objects:
        # Boost CLIP (better for object recognition)
        base_weights["gclip"] += 0.1
        if base_weights["asr"] > 0:
            base_weights["asr"] *= 0.8
    
    # Normalize weights
    total = sum(base_weights.values())
    return {k: v/total for k, v in base_weights.items()}

def create_frame_fusion_vector(asr_emb: np.ndarray, ocr_emb: np.ndarray, scene_emb: np.ndarray, 
                             gclip_emb: np.ndarray, weights: Dict[str, float], target_dim: int = 1024) -> np.ndarray:
    """Create fused frame vector"""
    # Note: Individual embeddings should already be L2-normalized from their respective functions
    # No need to normalize again here to avoid double normalization
    
    # Weighted concatenation
    fusion_vector = np.concatenate([
        weights["asr"] * asr_emb,      # 768
        weights["ocr"] * ocr_emb,      # 768  
        weights["scene"] * scene_emb,  # 768
        weights["gclip"] * gclip_emb   # 512
    ])
    
    # Project to target dimension using simple linear projection
    # In production, you might want to use learned projection or PCA
    current_dim = fusion_vector.shape[0]  # 768+768+768+512 = 2816
    if current_dim != target_dim:
        # Simple downsampling (in practice, use proper dimensionality reduction)
        step = current_dim // target_dim
        fusion_vector = fusion_vector[::step][:target_dim]
        
        # Pad if needed
        if len(fusion_vector) < target_dim:
            padding = np.zeros(target_dim - len(fusion_vector))
            fusion_vector = np.concatenate([fusion_vector, padding])
    
    # Final L2-normalization after fusion
    return l2_normalize(fusion_vector)

# ------------ 6) Main Processing Pipeline ------------
def process_video(video_id: str, data_root: str, output_root: str, models: Dict, 
                 use_asr: bool = False, use_scene: bool = False, use_ocr: bool = True, use_image: bool = True):
    """Process single video to generate embeddings
    
    Args:
        video_id: Video ID to process
        data_root: Root directory containing data
        output_root: Output directory for embeddings
        models: Models dictionary from init_models()
        use_asr: Whether to include ASR embedding (default: False)
        use_scene: Whether to include scene embedding (default: False) 
        use_ocr: Whether to include OCR embedding (default: True)
        use_image: Whether to include image embedding (default: True)
    """
    
    print(f"\n=== Processing video: {video_id} ===")
    
    # Create output directories
    frame_embed_dir = Path(output_root) / "embeds" / "frame_fusion" / video_id
    obj_embed_dir = Path(output_root) / "embeds" / "obj_crop" / video_id  
    frame_meta_dir = Path(output_root) / "meta" / "frame_meta" / video_id
    obj_meta_dir = Path(output_root) / "meta" / "obj_meta" / video_id
    
    for dir_path in [frame_embed_dir, obj_embed_dir, frame_meta_dir, obj_meta_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading metadata...")
    frame_map = load_frame_index(f"{data_root}/keyframes/{video_id}.csv")
    asr_data = load_asr_data(f"{data_root}/ASR/{video_id}.json")
    scene_map = load_scene_data(f"{data_root}/scene_recognize/{video_id}.json")
    
    asr_segments = asr_data.get("segments", [])
    
    # Process each frame
    frame_results = []
    obj_results = []
    
    print("Processing frames...")
    for frame_name, frame_info in tqdm(frame_map.items()):
        frame_idx = frame_info["frame_idx"]
        pts_time = frame_info["pts_time"]
        
        # Generate UIDs
        frame_uid = f"{video_id}:{frame_idx}"
        
        # Paths
        # frame_name already includes the zero-padded filename with extension, e.g. "001.jpg"
        image_path = f"{data_root}/frames/{video_id}/{frame_name}"
        stem = Path(frame_name).stem
        ocr_json_path = f"{data_root}/OCR/{video_id}/{stem}.json"
        obj_json_path = f"{data_root}/obj_detection/{video_id}/{stem}.json"
        
        # 1) Get ASR text
        asr_text, asr_conf = asr_text_for_frame(asr_segments, pts_time, slack=0.5)
        
        # 2) Get scene label
        scene_info = scene_map.get(frame_name, {})
        scene_text = scene_info.get("label_vi", "")
        scene_score = scene_info.get("score", 0.0)
        
        # 3) Get OCR data
        ocr_data = load_ocr_data(ocr_json_path)
        ocr_texts = ocr_data["texts"]
        ocr_bboxes = ocr_data["bboxes"] 
        ocr_conf = np.mean(ocr_data["confidences"]) if ocr_data["confidences"] else 0.0
        
        # 4) Get objects
        objects = load_object_data(obj_json_path)
        
        # === Generate embeddings ===
        
        # Generate embeddings based on selected modalities
        asr_emb = None
        ocr_emb = None  
        scene_emb = None
        gclip_emb = None
        
        if use_asr:
            asr_emb = embed_vietnamese_text(asr_text, models)
        else:
            asr_emb = np.zeros(768, dtype=np.float32)  # Zero vector if not used
        
        if use_ocr:
            ocr_emb = embed_ocr_layoutlmv3(image_path, ocr_texts, ocr_bboxes, models)
        else:
            ocr_emb = np.zeros(768, dtype=np.float32)  # Zero vector if not used
        
        if use_scene:
            weighted_scene_text = scene_text if scene_score > 0.3 else ""
            scene_emb = embed_vietnamese_text(weighted_scene_text, models) * scene_score
        else:
            scene_emb = np.zeros(768, dtype=np.float32)  # Zero vector if not used
        
        if use_image:
            gclip_emb = embed_clip_image(image_path, models)
        else:
            gclip_emb = np.zeros(512, dtype=np.float32)  # Zero vector if not used
        
        # Compute adaptive weights based on selected modalities
        weights = compute_adaptive_weights(
            asr_conf if use_asr else 0.0, 
            ocr_conf if use_ocr else 0.0, 
            scene_score if use_scene else 0.0, 
            len(objects) > 0
        )
        
        # Create fusion vector
        fusion_vector = create_frame_fusion_vector(asr_emb, ocr_emb, scene_emb, gclip_emb, weights)
        
        # Save frame embedding
        frame_embed_path = frame_embed_dir / f"{frame_idx}.npy"
        np.save(frame_embed_path, fusion_vector)
        
        # Save frame metadata
        frame_meta = {
            "uid": frame_uid,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "asr_text": asr_text,
            "asr_conf": float(asr_conf),
            "scene_label": scene_text,
            "scene_score": float(scene_score),
            "ocr_texts": ocr_texts,  # Add OCR texts for debugging and pre-filtering
            "ocr_tokens_kept": len(ocr_texts),
            "paths": {
                "image": image_path,
                "ocr_json": ocr_json_path,
                "obj_json": obj_json_path
            },
            "weights": {k: float(v) for k, v in weights.items()},
            "dims": {"frame": len(fusion_vector), "asr": len(asr_emb), "ocr": len(ocr_emb), "scene": len(scene_emb)}
        }
        
        frame_meta_path = frame_meta_dir / f"{frame_idx}.json"
        with open(frame_meta_path, 'w', encoding='utf-8') as f:
            json.dump(frame_meta, f, ensure_ascii=False, indent=2)
        
        frame_results.append(frame_meta)
        
        # === Process objects in this frame ===
        for obj in objects:
            obj_id = obj["obj_id"]
            obj_uid = f"{video_id}:{frame_idx}:{obj_id}"
            crop_path = obj["crop_path"]
            
            # If crop image missing, still allow text-only embedding path
            crop_available = os.path.exists(crop_path)
                
            # Calculate bbox area (normalized)
            bbox = obj["bbox_pixel"]
            if len(bbox) == 4:
                area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (1920 * 1080)  # assume 1080p
            else:
                area = 0.05  # default
            
            # Generate object embedding
            if crop_available:
                obj_emb = embed_obj_crop(crop_path, obj["label_vi"], models, area)
            else:
                # Fallback: text-only embedding to avoid skipping all objects
                obj_emb = embed_clip_text(obj["label_vi"], models)
                #obj_emb = l2_normalize(obj_emb)
            obj_emb = l2_normalize(obj_emb)
            
            # Save object embedding
            if crop_available:
                obj_embed_path = obj_embed_dir / f"{frame_idx}__{obj_id}.npy"
                np.save(obj_embed_path, obj_emb)
            
            # Save object metadata
            obj_meta = {
                "uid": obj_uid,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "label_vi": obj["label_vi"],
                "confidence": float(obj["confidence"]),
                "bbox_pixel": obj["bbox_pixel"],
                "crop_path": crop_path,
                "crop_available": crop_available,
                "dims": {"obj_embed": len(obj_emb)}
            }
            
            obj_meta_path = obj_meta_dir / f"{frame_idx}__{obj_id}.json"
            with open(obj_meta_path, 'w', encoding='utf-8') as f:
                json.dump(obj_meta, f, ensure_ascii=False, indent=2)
            
            obj_results.append(obj_meta)
    
    print(f"✅ Processed {len(frame_results)} frames, {len(obj_results)} objects for {video_id}")
    return frame_results, obj_results

# ------------ 7) Main Execution ------------
def main():
    # Configuration
    video_ids = ["L21_V001"]  # Add your video IDs
    data_root = "data"  # Root directory containing your data
    output_root = "data/embeddings"
    
    # Initialize models
    print("Initializing models...")
    models = init_models()
    
    # Process each video
    all_frame_results = []
    all_obj_results = []
    
    for video_id in video_ids:
        try:
            frame_results, obj_results = process_video(video_id, data_root, output_root, models)
            all_frame_results.extend(frame_results)
            all_obj_results.extend(obj_results)
        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            continue
    
    # Save global indexes
    print(f"\n=== Saving global indexes ===")
    
    # Frame index
    frame_index_path = Path(output_root) / "frame_index.json"
    with open(frame_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_frame_results, f, ensure_ascii=False, indent=2)
    
    # Object index  
    obj_index_path = Path(output_root) / "obj_index.json"
    with open(obj_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_obj_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Pipeline complete!")
    print(f"📊 Total: {len(all_frame_results)} frames, {len(all_obj_results)} objects")
    print(f"📁 Output directory: {output_root}")

if __name__ == "__main__":
    main()