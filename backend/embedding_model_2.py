"""
Embedding Model 2 - Image Only (CLIP-based)
==========================================

This module provides CLIP-only embedding functionality for video frames and objects.
It removes ASR, OCR, and scene modalities to minimize noise and focus on visual content.

Key differences from embedding_model.py:
- Frame embeddings: Only CLIP image embedding (512d)
- Text queries: Only CLIP text embedding (512d) 
- No fusion, no multimodal complexity
- Simplified and cleaner approach
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------ 1) Model Initialization ------------
def init_models() -> Dict:
    """Initialize CLIP models for image and text embedding"""
    try:
        import open_clip
        
        logger.info("Initializing CLIP models...")
        
        # Load CLIP model
        model_name = "ViT-B-32"
        pretrained = "openai"
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device,
            force_quick_gelu=True
        )
        
        tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move to device (in case it wasn't moved during creation)
        model = model.to(device)
        model.eval()
        
        logger.info(f"CLIP model loaded on {device}")
        
        return {
            'clip_model': model,
            'clip_preprocess': preprocess,
            'clip_tokenizer': tokenizer,
            'device': device
        }
        
    except ImportError:
        logger.error("open_clip not installed. Install with: pip install open_clip_torch")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize CLIP models: {e}")
        raise

# ------------ 2) Image Embedding Functions ------------
def embed_clip_image(image_path: str, models: Dict) -> np.ndarray:
    """Embed image using CLIP"""
    try:
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = models['clip_preprocess'](image).unsqueeze(0).to(models['device'])
        
        # Get embedding
        with torch.no_grad():
            image_features = models['clip_model'].encode_image(image_tensor)
            image_features = image_features.cpu().numpy().flatten().astype(np.float32)
            
            # L2-normalize the embedding
            norm = np.linalg.norm(image_features)
            if norm > 1e-12:
                image_features = image_features / norm
            
            return image_features
        
    except Exception as e:
        logger.error(f"Error in CLIP image embedding: {e}")
        # Return zero vector as fallback
        return np.zeros(512, dtype=np.float32)

def embed_obj_crop(crop_path: str, label: str, models: Dict, area: float = 0.05) -> np.ndarray:
    """Embed object crop using CLIP with label context"""
    try:
        from PIL import Image
        
        # Load crop image
        if not os.path.exists(crop_path):
            # Fallback to text-only embedding
            return embed_clip_text(label, models)
        
        image = Image.open(crop_path).convert('RGB')
        image_tensor = models['clip_preprocess'](image).unsqueeze(0).to(models['device'])
        
        # Get image embedding
        with torch.no_grad():
            image_features = models['clip_model'].encode_image(image_tensor)
            image_features = image_features.cpu().numpy().flatten().astype(np.float32)
            
            # L2-normalize the embedding
            norm = np.linalg.norm(image_features)
            if norm > 1e-12:
                image_features = image_features / norm
            
            return image_features
        
    except Exception as e:
        logger.error(f"Error in object crop embedding: {e}")
        # Fallback to text-only embedding
        return embed_clip_text(label, models)

# ------------ 3) Text Embedding Functions ------------
def embed_clip_text(text: str, models: Dict) -> np.ndarray:
    """Embed text using CLIP"""
    if not text.strip():
        return np.zeros(512, dtype=np.float32)
    
    try:
        clip_model = models['clip_model']
        tokenizer = models['clip_tokenizer']
        
        # Tokenize
        text_tokens = tokenizer([text]).to(models['device'])
        
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

# ------------ 4) Normalization ------------
def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vector"""
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return vector / norm

# ------------ 5) Data Loading Functions ------------
def load_asr_data(json_path: str) -> Dict[str, Any]:
    """Load ASR segments JSON: expects {"segments": [{"start","end","text",...}, ...]}"""
    try:
        if not os.path.exists(json_path):
            return {"segments": []}
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ASR data: {e}")
        return {"segments": []}

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

def load_object_data(json_path: str) -> List[Dict]:
    """Load object detection data"""
    try:
        if not os.path.exists(json_path):
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        objects = []
        for i, obj in enumerate(data.get('objects', [])):
            objects.append({
                'obj_id': obj.get('id', f"obj_{i}"),
                'label_vi': obj.get('label_vi', ''),
                'confidence': float(obj.get('confidence', 0)),
                'bbox_pixel': obj.get('bbox', []),
                'crop_path': obj.get('crop_path', '')
            })
        return objects
    except Exception as e:
        logger.error(f"Error loading object data: {e}")
        return []

# ------------ 6) Main Processing Pipeline ------------
def process_video(video_id: str, data_root: str, output_root: str, models: Dict):
    """Process single video to generate CLIP-only embeddings"""
    
    print(f"\n=== Processing video: {video_id} (CLIP-only) ===")
    
    # Create output directories
    frame_embed_dir = Path(output_root) / "embeds" / "frame_clip" / video_id
    obj_embed_dir = Path(output_root) / "embeds" / "obj_crop" / video_id  
    frame_meta_dir = Path(output_root) / "meta" / "frame_meta" / video_id
    obj_meta_dir = Path(output_root) / "meta" / "obj_meta" / video_id
    
    for dir_path in [frame_embed_dir, obj_embed_dir, frame_meta_dir, obj_meta_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load frame index
    print("Loading frame/ASR/scene indexes...")
    frame_map = load_frame_index(f"{data_root}/keyframes/{video_id}.csv")
    asr_data = load_asr_data(f"{data_root}/ASR/{video_id}.json")
    asr_segments = asr_data.get("segments", [])
    scene_map = load_scene_data(f"{data_root}/scene_recognize/{video_id}.json")
    
    # Process each frame
    frame_results = []
    obj_results = []
    
    print("Processing frames...")
    for frame_name, frame_info in tqdm(frame_map.items()):
        frame_idx = frame_info["frame_idx"]
        pts_time = frame_info["pts_time"]
        
        # Generate UID
        frame_uid = f"{video_id}:{frame_idx}"
        
        # Image path
        image_path = f"{data_root}/frames/{video_id}/{frame_name}"
        
        # Object/OCR data paths
        stem = Path(frame_name).stem
        obj_json_path = f"{data_root}/obj_detection/{video_id}/{stem}.json"
        ocr_json_path = f"{data_root}/OCR/{video_id}/{stem}.json"

        # Populate metadata fields similar to embedding_model
        asr_text, _ = asr_text_for_frame(asr_segments, pts_time, slack=0.5)
        scene_info = scene_map.get(frame_name, {}) if isinstance(scene_map, dict) else {}
        scene_text = str(scene_info.get("label_vi", ""))
        ocr_bundle = load_ocr_data(ocr_json_path)
        ocr_texts: List[str] = ocr_bundle.get("texts", [])
        
        # Generate CLIP image embedding
        clip_emb = embed_clip_image(image_path, models)
        # Don't normalize here - will be normalized during index building
        
        # Save frame embedding (512d CLIP only)
        frame_embed_path = frame_embed_dir / f"{frame_idx}.npy"
        np.save(frame_embed_path, clip_emb)
        
        # Save frame metadata (compatible format)
        frame_meta = {
            "uid": frame_uid,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "asr_text": asr_text,
            "scene_label": scene_text,
            "ocr_texts": ocr_texts,
            "paths": {
                "image": image_path,
                "ocr_json": ocr_json_path,
                "obj_json": obj_json_path
            },
            "embedding_type": "clip_image_only",
            "dims": {"frame": len(clip_emb)}
        }
        
        frame_meta_path = frame_meta_dir / f"{frame_idx}.json"
        with open(frame_meta_path, 'w', encoding='utf-8') as f:
            json.dump(frame_meta, f, ensure_ascii=False, indent=2)
        
        frame_results.append(frame_meta)
        
        # Process objects (unchanged from original)
        objects = load_object_data(obj_json_path)
        for obj in objects:
            obj_id = obj["obj_id"]
            obj_uid = f"{video_id}:{frame_idx}:{obj_id}"
            crop_path = obj["crop_path"]
            
            # Check if crop image exists
            crop_available = os.path.exists(crop_path)
            
            # Calculate bbox area (normalized)
            bbox = obj["bbox_pixel"]
            if len(bbox) == 4:
                area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (1920 * 1080)
            else:
                area = 0.05
            
            # Generate object embedding
            if crop_available:
                obj_emb = embed_obj_crop(crop_path, obj["label_vi"], models, area)
            else:
                # Fallback to text-only embedding
                obj_emb = embed_clip_text(obj["label_vi"], models)
            # Don't normalize here - will be normalized during index building
            
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
    output_root = "data/embeddings_v2"  # New output directory
    
    # Initialize models
    print("Initializing CLIP models...")
    models = init_models()
    all_frame_results = []
    all_obj_results = []
    # Process videos
    for video_id in video_ids:
        try:
            frame_results, obj_results = process_video(video_id, data_root, output_root, models)
            all_frame_results.extend(frame_results)
            all_obj_results.extend(obj_results)
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            continue
    
    print("✅ All videos processed successfully!")
    # Frame index
    
    frame_index_path = Path(output_root) / "frame_index.json"
    with open(frame_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_frame_results, f, ensure_ascii=False, indent=2)
    
    # Object index  
    obj_index_path = Path(output_root) / "obj_index.json"
    with open(obj_index_path, 'w', encoding='utf-8') as f:
        json.dump(all_obj_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved indexes to {output_root}")
if __name__ == "__main__":
    main()
