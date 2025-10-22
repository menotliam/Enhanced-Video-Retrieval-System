"""
Object detector using YOLOv8 for fallback detection when preprocessed files are missing
"""
import os
import json
import csv
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
from loguru import logger

#from app.config import settings, MODEL_CONFIG


class ObjectDetector:
    """YOLOv8-based object detector for fallback detection"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.36
        self.max_objects = 40
        
        # Vietnamese class name mapping for common COCO classes
        self.vn_class_mapping = {
            "person": "người",
            "bicycle": "xe đạp", 
            "car": "xe hơi",
            "motorcycle": "xe máy",
            "airplane": "máy bay",
            "bus": "xe buýt",
            "train": "tàu hỏa",
            "truck": "xe tải",
            "boat": "thuyền",
            "traffic light": "đèn giao thông",
            "fire hydrant": "vòi cứu hỏa",
            "stop sign": "biển báo dừng",
            "parking meter": "đồng hồ đỗ xe",
            "bench": "ghế dài",
            "bird": "chim",
            "cat": "mèo",
            "dog": "chó",
            "horse": "ngựa",
            "sheep": "cừu",
            "cow": "bò",
            "elephant": "voi",
            "bear": "gấu",
            "zebra": "ngựa vằn",
            "giraffe": "hươu cao cổ",
            "backpack": "ba lô",
            "umbrella": "ô",
            "handbag": "túi xách",
            "tie": "cà vạt",
            "suitcase": "vali",
            "frisbee": "đĩa bay",
            "skis": "ván trượt tuyết",
            "snowboard": "ván trượt tuyết",
            "sports ball": "bóng thể thao",
            "kite": "diều",
            "baseball bat": "gậy bóng chày",
            "baseball glove": "găng tay bóng chày",
            "skateboard": "ván trượt",
            "surfboard": "ván lướt sóng",
            "tennis racket": "vợt tennis",
            "bottle": "chai",
            "wine glass": "ly rượu",
            "cup": "cốc",
            
            "knife": "dao",
            "spoon": "thìa",
            "bowl": "bát",
            "banana": "chuối",
            "apple": "táo",
            "sandwich": "bánh mì",
            "orange": "cam",
            "broccoli": "bông cải xanh",
            "carrot": "cà rốt",
            "hot dog": "hot dog",
            "pizza": "pizza",
            "donut": "bánh donut",
            "cake": "bánh ngọt",
            "chair": "ghế",
            "couch": "ghế sofa",
            "potted plant": "cây cảnh",
            "bed": "giường",
            "dining table": "bàn ăn",
            "toilet": "toilet",
            "tv": "tivi",
            "laptop": "laptop",
            "mouse": "chuột máy tính",
            "remote": "điều khiển từ xa",
            "keyboard": "bàn phím",
            "cell phone": "điện thoại",
            "microwave": "lò vi sóng",
            "oven": "lò nướng",
            "toaster": "máy nướng bánh",
            "sink": "bồn rửa",
            "refrigerator": "tủ lạnh",
            "book": "sách",
            "clock": "đồng hồ",
            "vase": "lọ hoa",
            "scissors": "kéo",
            "teddy bear": "gấu bông",
            "hair drier": "máy sấy tóc",
            "toothbrush": "bàn chải đánh răng",
            "poster": "bảng",
            "sign": "biển báo",
            "traffic sign": "biển báo giao thông"
            
        }
        
        logger.info(f"ObjectDetector initialized on device: {self.device}")

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            logger.info("Loading YOLOv8 model...")
            
            # Try to import ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics not installed. Install with: pip install ultralytics")
                raise
            
            # Load model
            model_path = "yolov8l.pt"
            logger.info(f"Loading YOLO model: {model_path}")
            
            self.model = YOLO(model_path)
            
            # Move to device if possible
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            logger.info("YOLOv8 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
            raise

    async def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect objects in image and return YOLO-compatible format
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detection results in YOLO JSON format
        """
        if self.model is None:
            logger.warning("YOLO model not loaded")
            return self._empty_detection_result()
        
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return self._empty_detection_result()
            
            logger.debug(f"Running object detection on: {image_path}")
            
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return self._empty_detection_result()
            
            # Extract first result (single image)
            result = results[0]
            
            # Extract detection data
            detection_data = self._extract_detection_data(result)
            
            logger.debug(f"Detected {len(detection_data['detection_scores'])} objects")
            
            return detection_data
            
        except Exception as e:
            logger.error(f"Object detection failed for {image_path}: {str(e)}")
            return self._empty_detection_result()

    def _extract_detection_data(self, result) -> Dict[str, Any]:
        """Extract detection data from YOLO result in required format"""
        detection_scores = []
        detection_class_names = []
        detection_class_entities = []
        detection_boxes = []
        detection_class_labels = []
        
        if result.boxes is not None:
            # Get boxes, scores, and class IDs
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Limit number of objects
            if len(boxes) > self.max_objects:
                # Sort by confidence and keep top N
                indices = np.argsort(scores)[::-1][:self.max_objects]
                boxes = boxes[indices]
                scores = scores[indices]
                class_ids = class_ids[indices]
            
            # Extract data for each detection
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= self.confidence_threshold:
                    # Get class name
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # Get Vietnamese translation
                    vn_class_name = self.vn_class_mapping.get(class_name, class_name)
                    
                    detection_scores.append(float(score))
                    detection_class_names.append(class_name)
                    detection_class_entities.append(vn_class_name)
                    detection_boxes.append([float(x) for x in box])  # [x1, y1, x2, y2]
                    detection_class_labels.append(int(class_id))
        
        return {
            "detection_scores": detection_scores,
            "detection_class_names": detection_class_names,
            "detection_class_entities": detection_class_entities,
            "detection_boxes": detection_boxes,
            "detection_class_labels": detection_class_labels
        }

    def _empty_detection_result(self) -> Dict[str, Any]:
        """Return empty detection result"""
        return {
            "detection_scores": [],
            "detection_class_names": [],
            "detection_class_entities": [],
            "detection_boxes": [],
            "detection_class_labels": []
        }

    def detect_and_save(self, image_path: str, output_path: str) -> bool:
        """
        Detect objects and save results to JSON file (legacy schema)
        
        Args:
            image_path: Path to input image
            output_path: Path to save JSON results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Run detection
            import asyncio
            detection_result = asyncio.run(self.detect(image_path))
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(detection_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detection results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to detect and save: {str(e)}")
            return False

    def _load_keyframe_id_map(self, video_id: str) -> Dict[str, int]:
        """Load mapping from visual frame name (e.g., '001') to CSV frame_idx.
        CSV path: data/keyframes/<video_id>.csv with columns: n, pts_time, fps, frame_idx
        Returns mapping like {'001': 0, '002': 90, ...}
        """
        mapping: Dict[str, int] = {}
        csv_path = os.path.join('data', 'keyframes', f"{video_id}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"Keyframes CSV not found: {csv_path}. Fallback to filename-based ids.")
            return mapping
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # 'n' is 1-based visual frame number → stem '001', '002', ...
                        n_val = int(float(row.get('n')))
                        stem = f"{n_val:03d}"
                        # frame_idx is the absolute frame index in the video timeline
                        frame_idx_val = int(float(row.get('frame_idx')))
                        mapping[stem] = frame_idx_val
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Failed to read keyframes CSV {csv_path}: {e}")
        # Log a small sample for verification
        sample = dict(list(mapping.items())[:5])
        logger.info(f"Keyframe ID map loaded (count={len(mapping)}), sample={sample}")
        return mapping

    def _crop_object(self, image_path: str, bbox: List[float], crops_dir: str, frame_stem: str, obj_idx: int) -> str:
        """Crop object from image by bbox [x1,y1,x2,y2] and save to crops_dir.
        Returns relative path from backend root to saved crop.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            # clamp
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid bbox after clamping")

            os.makedirs(crops_dir, exist_ok=True)
            crop = img.crop((x1, y1, x2, y2))
            crop_filename = f"{frame_stem}_obj_{obj_idx}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            crop.save(crop_path, format='JPEG', quality=95, subsampling=0, optimize=True)
            # Return normalized relative path
            return crop_path.replace('\\', '/')
        except Exception as e:
            logger.error(f"Failed to crop object for {image_path}: {e}")
            return ""

    def detect_and_save_new_schema(self, image_path: str, output_path: str, video_id: str, keyframe_id_map: Optional[Dict[str, int]] = None) -> bool:
        """
        Detect objects and save results to JSON using new optimized schema:
        {
          "frame_id": <int>,
          "objects": [
            {
              "id": "obj_<idx>",
              "label_vi": <str>,
              "confidence": <float>,
              "bbox": [x1,y1,x2,y2],
              "crop_path": "data/frames/<video_id>/crops/<frame>_obj<idx>.jpg",
              "embedding_id": "vec_<video_id>_<frame_id>_<idx>"
            }
          ]
        }
        """
        try:
            import asyncio
            raw = asyncio.run(self.detect(image_path))

            # Resolve frame_stem and frame_id
            frame_stem = Path(image_path).stem
            if keyframe_id_map and frame_stem in keyframe_id_map:
                frame_id = keyframe_id_map[frame_stem]
            else:
                # Fallback to numeric stem
                try:
                    frame_id = int(frame_stem)
                except Exception:
                    frame_id = 0

            # Prepare crops dir: data/frames/<video_id>/crops/
            crops_dir = os.path.join('data', 'obj_det', video_id)

            objects: List[Dict[str, Any]] = []
            scores = raw.get('detection_scores', [])
            labels_vi = raw.get('detection_class_entities', [])
            boxes = raw.get('detection_boxes', [])

            for idx, (score, label_vi, bbox) in enumerate(zip(scores, labels_vi, boxes), start=1):
                crop_path = self._crop_object(image_path, bbox, crops_dir, frame_stem, idx)
                embedding_id = f"vec_{video_id}_{frame_id}_{idx}"
                obj_entry = {
                    "id": f"obj_{idx}",
                    "label_vi": label_vi,
                    "confidence": float(score),
                    "bbox": [int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))],
                    "crop_path": crop_path,
                    "embedding_id": embedding_id
                }
                objects.append(obj_entry)

            result = {
                "frame_id": int(frame_id),
                "objects": objects
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"New-schema detection saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to detect and save (new schema) for {image_path}: {e}")
            return False

    def batch_detect(self, image_paths: List[str], output_dir: str, use_new_schema: bool = False, video_id: Optional[str] = None) -> Dict[str, bool]:
        """
        Run detection on multiple images and save results
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save JSON results
            use_new_schema: If True, output in optimized schema and save crops
            video_id: Required when use_new_schema=True to build crop and embedding ids
            
        Returns:
            Dictionary mapping image paths to success status
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        keyframe_id_map: Dict[str, int] = {}
        if use_new_schema:
            if not video_id:
                # Try to infer from output_dir (expects data/obj_detection/<video_id>)
                video_id = Path(output_dir).name
            keyframe_id_map = self._load_keyframe_id_map(video_id)
        
        for image_path in image_paths:
            try:
                # Generate output filename
                image_name = Path(image_path).stem
                json_filename = f"{image_name}.json"
                json_path = output_path / json_filename
                
                # Run detection and save
                if use_new_schema:
                    success = self.detect_and_save_new_schema(image_path, str(json_path), video_id=video_id or "", keyframe_id_map=keyframe_id_map)
                else:
                    success = self.detect_and_save(image_path, str(json_path))
                results[image_path] = success
                
            except Exception as e:
                logger.error(f"Batch detection failed for {image_path}: {str(e)}")
                results[image_path] = False
        
        return results

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_loaded": self.is_loaded(),
            "model_path": "yolov8l",
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
            "max_objects": self.max_objects,
            "supported_classes": len(self.vn_class_mapping)
        }

    def get_supported_classes(self) -> Dict[str, str]:
        """Get mapping of English to Vietnamese class names"""
        return self.vn_class_mapping.copy()


# Global object detector instance
object_detector = ObjectDetector()