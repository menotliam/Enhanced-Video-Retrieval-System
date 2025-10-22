"""
Visual pipeline: keyframe extraction, object detection, OCR, scene classification, visual embedding.
Input: Scene video segment or preprocessed frames
Output: List of frame dicts with embeddings and metadata
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from loguru import logger

from app.config import settings
from app.models.embedding_model import embedding_model
from app.models.object_detector import object_detector
from app.models.ocr_model import OCRModel


class VisualPipeline:
    """Visual processing pipeline for video frames"""
    
    def __init__(self):
        self.keyframes_per_scene = settings.KEYFRAMES_PER_SCENE
        logger.info("VisualPipeline initialized")
        self.ocr_model = None
        try:
            self.ocr_model = OCRModel()
            logger.info("OCR model initialized successfully")
        except Exception as e:
            logger.warning(f"OCR model initialization failed: {e}")

    async def process_scene(
        self, 
        scene_video_path: Optional[str] = None, 
        video_id: str = "",
        frames_dir: Optional[str] = None,
        frame_indices: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a scene: extract keyframes, detect objects, OCR, classify scene, embed frames.
        
        Args:
            scene_video_path: Path to scene video file (optional)
            scene_id: Unique scene identifier
            frames_dir: Directory containing preprocessed frames
            frame_indices: List of frame indices to process
            
        Returns: List of dicts for each frame with embeddings and metadata
        """
        try:
            if frames_dir and frame_indices:
                # Process preprocessed frames
                return await self._process_preprocessed_frames(frames_dir, frame_indices, video_id)
            elif scene_video_path:
                # Extract and process keyframes from video
                return await self._process_video_scene(scene_video_path, video_id)
            else:
                logger.error("Either frames_dir+frame_indices or scene_video_path must be provided")
                return []
                
        except Exception as e:
            logger.error(f"Scene processing failed for {video_id}: {str(e)}")
            return []

    async def _process_preprocessed_frames(
        self, 
        frames_dir: str, 
        frame_indices: List[int], 
        video_id: str
    ) -> List[Dict[str, Any]]:
        """Process preprocessed frames from directory"""
        frame_results = []
        
        for frame_idx in frame_indices:
            try:
                # Find frame file
                frame_path = self._find_frame_file(frames_dir, frame_idx)
                if not frame_path:
                    logger.warning(f"Frame {frame_idx} not found in {frames_dir}")
                    continue
                
                # Process individual frame
                frame_result = await self._process_single_frame(
                    frame_path, frame_idx, video_id
                )
                
                if frame_result:
                    frame_results.append(frame_result)
                    
            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(frame_results)} frames for scene {video_id}")
        return frame_results

    async def _process_video_scene(self, scene_video_path: str, video_id: str) -> List[Dict[str, Any]]:
        """Extract keyframes from video and process them"""
        try:
            # Extract keyframes
            keyframes = self._extract_keyframes(scene_video_path)
            
            frame_results = []
            for i, (frame, timestamp) in enumerate(keyframes):
                try:
                    # Save frame temporarily
                    temp_frame_path = f"/tmp/{video_id}_frame_{i}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Process frame
                    frame_result = await self._process_single_frame(
                        temp_frame_path, i, video_id, timestamp
                    )
                    
                    if frame_result:
                        frame_results.append(frame_result)
                    
                    # Clean up
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                        
                except Exception as e:
                    logger.error(f"Failed to process keyframe {i}: {str(e)}")
                    continue
            
            return frame_results
            
        except Exception as e:
            logger.error(f"Video scene processing failed: {str(e)}")
            return []

    async def _process_single_frame(
        self, 
        frame_path: str, 
        frame_idx: int, 
        video_id: str,
        timestamp: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Process a single frame with all visual analysis"""
        try:
            # Create frame ID
            frame_id = f"{video_id}_frame_{frame_idx}"
            
            # Read frame
            if not os.path.exists(frame_path):
                logger.error(f"Frame file not found: {frame_path}")
                return None
            
            # Object detection
            detection_result = await self._detect_objects(frame_path)
            detected_objects = detection_result.get("detection_class_entities", [])
            
            # OCR text extraction
            ocr_text = await self._extract_text(frame_path)
            
            # Scene classification (simplified)
            scene_label = await self._classify_scene(frame_path)
            
            # Generate visual embedding
            visual_embedding = await embedding_model.encode_image(frame_path)
            
            # Build frame result
            frame_result = {
                "frame_id": frame_id,
                "frame_idx": frame_idx,
                "video_id": video_id,
                "timestamp": timestamp or 0.0,
                "file_path": frame_path,
                "detected_objects": detected_objects,
                "detection_data": detection_result,  # Full YOLO detection data
                "ocr_text": ocr_text,
                "scene_label": scene_label,
                "visual_embedding": visual_embedding,
                "metadata": {
                    "width": None,
                    "height": None,
                    "channels": None
                }
            }
            
            # Get image dimensions
            try:
                image = Image.open(frame_path)
                frame_result["metadata"]["width"] = image.width
                frame_result["metadata"]["height"] = image.height
                frame_result["metadata"]["channels"] = len(image.getbands())
            except Exception:
                pass
            
            return frame_result
            
        except Exception as e:
            logger.error(f"Single frame processing failed: {str(e)}")
            return None

    def _find_frame_file(self, frames_dir: str, frame_idx: int) -> Optional[str]:
        """Find frame file by index in frames directory"""
        # Common frame file patterns
        patterns = [
            f"{frame_idx:03d}.jpg",
            f"{frame_idx:03d}.png", 
            f"{frame_idx}.jpg",
            f"{frame_idx}.png",
            f"frame_{frame_idx:03d}.jpg",
            f"frame_{frame_idx:03d}.png"
        ]
        
        for pattern in patterns:
            frame_path = os.path.join(frames_dir, pattern)
            if os.path.exists(frame_path):
                return frame_path
        
        return None

    def _extract_keyframes(self, video_path: str) -> List[tuple]:
        """Extract keyframes from video"""
        keyframes = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Extract frames at even intervals
            interval = max(1, frame_count // self.keyframes_per_scene)
            
            frame_indices = []
            for i in range(self.keyframes_per_scene):
                frame_idx = i * interval
                if frame_idx < frame_count:
                    frame_indices.append(frame_idx)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / fps if fps > 0 else 0
                    keyframes.append((frame, timestamp))
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Keyframe extraction failed: {str(e)}")
        
        return keyframes

    async def _detect_objects(self, frame_path: str) -> Dict[str, Any]:
        """Detect objects in frame using YOLO fallback"""
        try:
            # Use object detector if available and loaded
            if object_detector.is_loaded():
                detections = await object_detector.detect(frame_path)
                return detections
            else:
                logger.warning("Object detector not loaded, attempting to load...")
                try:
                    object_detector.load_model()
                    detections = await object_detector.detect(frame_path)
                    return detections
                except Exception as e:
                    logger.warning(f"Could not load object detector: {str(e)}")
                    return {
                        "detection_scores": [],
                        "detection_class_names": [],
                        "detection_class_entities": [],
                        "detection_boxes": [],
                        "detection_class_labels": []
                    }
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return {
                "detection_scores": [],
                "detection_class_names": [],
                "detection_class_entities": [],
                "detection_boxes": [],
                "detection_class_labels": []
            }

    async def _extract_text(self, frame_path: str) -> str:
        """Extract text from frame using OCR"""
        try:
            # Use OCR model if available
            if self.ocr_model is not None:
                ocr_results = self.ocr_model.extract_text(frame_path)
                
                if ocr_results:
                    extracted_texts = [result['text'] for result in ocr_results if result.get('text')]
                    return ' '.join(extracted_texts)
                return ""
            else:
                logger.warning("OCR model not available")
                return ""
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""

    async def _classify_scene(self, frame_path: str) -> str:
        """Classify scene type (simplified implementation)"""
        try:
            # Simple scene classification based on filename or basic image analysis
            # In a full implementation, you would use a scene classification model
            
            # For now, return a default classification
            return "outdoor"  # or "indoor", "street", "nature", etc.
            
        except Exception as e:
            logger.error(f"Scene classification failed: {str(e)}")
            return "unknown"

    async def process_batch_frames(
        self, 
        frames_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple frames in batch"""
        results = []
        
        for frame_data in frames_data:
            frame_path = frame_data.get("frame_path")
            frame_idx = frame_data.get("frame_idx", 0)
            scene_id = frame_data.get("scene_id", "")
            timestamp = frame_data.get("timestamp", 0.0)
            
            if frame_path:
                result = await self._process_single_frame(
                    frame_path, frame_idx, scene_id, timestamp
                )
                if result:
                    results.append(result)
        
        return results

    def get_supported_formats(self) -> List[str]:
        """Get supported image formats"""
        return [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


# Global visual pipeline instance
visual_pipeline = VisualPipeline()
