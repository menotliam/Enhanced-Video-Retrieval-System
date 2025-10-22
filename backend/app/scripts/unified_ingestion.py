"""
Unified ingestion script for AI Challenge 2025
Processes preprocessed video data (frames + metadata + YOLO) into the search system

Usage:
    python unified_ingestion.py --video-id L21_V001 --data-dir ./data
"""

import os
import sys
import json
import csv
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.db.database import SessionLocal, create_tables
from app.db.repository import VideoRepository, SceneRepository, FrameRepository
from app.db.vector_db import vector_db
from app.db.metadata_db import metadata_db
from app.models.embedding_model import embedding_model
from app.services.visual_pipeline import visual_pipeline
from app.services.audio_pipeline import audio_pipeline
from app.services.fusion import fusion_service
from app.config import settings
from app.utils.text_normalizer import text_normalizer


class UnifiedIngestion:
    """Unified ingestion pipeline for preprocessed video data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        self.keyframes_dir = self.data_dir / "keyframes"
        self.metadata_dir = self.data_dir / "metadata"
        self.obj_detection_dir = self.data_dir / "obj_detection"
        
        logger.info(f"UnifiedIngestion initialized with data_dir: {data_dir}")

    async def initialize_services(self):
        """Initialize all required services and models"""
        logger.info("Initializing services and models...")
        
        # Create database tables
        create_tables()
        
        # Initialize models
        embedding_model.load_models()
        
        # Initialize databases
        vector_db.initialize()
        metadata_db.initialize()
        
        logger.info("All services initialized successfully")

    async def ingest_video(self, video_id: str) -> Dict[str, Any]:
        """
        Ingest a preprocessed video into the system
        
        Args:
            video_id: Video identifier (e.g., "L21_V001")
            
        Returns:
            Ingestion results
        """
        logger.info(f"Starting ingestion for video: {video_id}")
        
        try:
            # Check data availability
            video_data = self._check_video_data(video_id)
            if not video_data["valid"]:
                return {"error": f"Invalid video data: {video_data['missing']}"}
            
            # Create/update video record
            db = SessionLocal()
            try:
                video_record = await self._create_video_record(db, video_id, video_data)
                
                # Process frames
                frame_results = await self._process_frames(video_id, video_data)
                
                # Index frames in databases
                indexing_results = await self._index_frames(frame_results)
                
                # Generate summary
                summary = self._generate_summary(video_id, frame_results, indexing_results)
                
                logger.info(f"Ingestion completed for {video_id}")
                return summary
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Ingestion failed for {video_id}: {str(e)}")
            return {"error": str(e)}

    def _check_video_data(self, video_id: str) -> Dict[str, Any]:
        """Check if required data files exist for the video"""
        required_files = {
            "keyframes_csv": self.keyframes_dir / f"{video_id}.csv",
            "video_frames_dir": self.frames_dir / video_id,
            "metadata_json": self.metadata_dir / f"{video_id}.json"
        }
        
        missing = []
        available = {}
        
        for key, path in required_files.items():
            if path.exists():
                available[key] = str(path)
            else:
                missing.append(str(path))
        
        # Check YOLO detection files
        yolo_dir = self.obj_detection_dir
        yolo_files = []
        if yolo_dir.exists():
            yolo_files = list(yolo_dir.glob("*.json"))
        
        available["yolo_files"] = [str(f) for f in yolo_files]
        
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "available": available
        }

    async def _create_video_record(self, db, video_id: str, video_data: Dict[str, Any]) -> Any:
        """Create or update video record in database"""
        video_repo = VideoRepository(db)
        
        # Load metadata if available
        metadata = {}
        metadata_path = video_data["available"].get("metadata_json")
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Create video data
        video_record_data = {
            "video_id": video_id,
            "filename": metadata.get("title", f"{video_id}.mp4"),
            "file_path": f"./data/videos/{video_id}.mp4",
            "duration": float(metadata.get("length", 0)),
            "format": ".mp4",
            "processed": True
        }
        
        # Create or update
        existing = video_repo.get_video_by_id(video_id)
        if existing:
            return video_repo.update_video(video_id, video_record_data)
        else:
            return video_repo.create_video(video_record_data)

    async def _process_frames(self, video_id: str, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all frames for the video"""
        logger.info(f"Processing frames for {video_id}")
        
        # Load keyframes CSV
        keyframes_csv = video_data["available"]["keyframes_csv"]
        frames_dir = video_data["available"]["video_frames_dir"]
        
        frames_info = self._load_keyframes_csv(keyframes_csv)
        logger.info(f"Loaded {len(frames_info)} frames from CSV")
        
        frame_results = []
        
        for i, frame_info in enumerate(frames_info):
            try:
                frame_idx = frame_info["frame_idx"]
                start_time = frame_info["start_time"]
                numeric_idx = frame_info["numeric_idx"]
                logger.info(f"Processing frame {i+1}/{len(frames_info)}: index={frame_idx}")
                
                # Find frame file
                frame_path = self._find_frame_file(frames_dir, numeric_idx)
                if not frame_path:
                    logger.warning(f"Frame {numeric_idx} not found in {frames_dir}")
                    continue
                
                # Process frame through visual pipeline
                #scene_id = f"{video_id}_frame_{frame_idx:03d}"
                visual_result = await visual_pipeline.process_scene(
                    frames_dir=frames_dir,
                    frame_indices=[numeric_idx],
                    video_id=video_id
                )
                
                if not visual_result:
                    logger.warning(f"Visual processing failed for frame {frame_idx}")
                    continue
                
                # Get YOLO detection data (with fallback to real-time detection)
                yolo_data = await self._load_yolo_data_with_fallback(
                    numeric_idx, 
                    video_data["available"]["yolo_files"],
                    frame_path
                )
                
                # Combine results
                frame_result = {
                    "video_id": video_id,
                    "numeric_idx": numeric_idx,
                    "frame_idx": frame_idx,
                    "start_time": start_time,
                    "end_time": start_time,  # Single frame
                    "duration": 0.0,
                    "frame_path": frame_path,
                    "visual_data": visual_result[0] if visual_result else {},
                    "yolo_data": yolo_data,
                    "audio_data": {},  # No audio for individual frames
                }
                
                frame_results.append(frame_result)
                
            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(frame_results)} frames")
        return frame_results

    def _load_keyframes_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load frame information from keyframes CSV"""
        frames = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats
                frame_idx = int(row.get("frame_idx"))
                numeric_idx = int(row.get("n"))
                # Calculate start time
                start_time_raw = row.get("pts_time")
                if start_time_raw:
                    start_time = float(start_time_raw)
                else:
                    # Calculate from frame index and FPS
                    n = int(row.get("frame_idx", 0))
                    fps = float(row.get("fps", 30.0))
                    start_time = n / fps if fps > 0 else 0.0
                
                frames.append({
                    "frame_idx": frame_idx,
                    "start_time": start_time,
                    "numeric_idx": numeric_idx
                })
        
        return frames

    def _find_frame_file(self, frames_dir: str, numeric_idx: int) -> Optional[str]:
        """Find frame file by index"""
        patterns = [
            f"{numeric_idx:03d}.jpg",
            f"{numeric_idx:03d}.png",
            f"{numeric_idx}.jpg",
            f"{numeric_idx}.png"
        ]
        
        for pattern in patterns:
            frame_path = os.path.join(frames_dir, pattern)
            if os.path.exists(frame_path):
                return frame_path
        
        return None

    def _load_yolo_data(self, numeric_idx: int, yolo_files: List[str]) -> Dict[str, Any]:
        """Load YOLO detection data for frame from preprocessed files"""
        # Find YOLO file for this frame
        yolo_file = None
        for file_path in yolo_files:
            filename = os.path.basename(file_path)
            if f"{numeric_idx:03d}.json" in filename or f"{numeric_idx}.json" in filename:
                yolo_file = file_path
                break
        
        if not yolo_file:
            return {}
        
        try:
            with open(yolo_file, 'r', encoding='utf-8') as f:
                yolo_data = json.load(f)
            
            # Extract object names
            detected_objects = []
            if "detection_class_entities" in yolo_data:
                detected_objects = [str(obj).lower() for obj in yolo_data["detection_class_entities"]]
            elif "detection_class_names" in yolo_data:
                detected_objects = [str(obj).lower() for obj in yolo_data["detection_class_names"]]
            
            return {
                "detected_objects": list(set(detected_objects)),
                "detection_scores": yolo_data.get("detection_scores", []),
                "detection_boxes": yolo_data.get("detection_boxes", []),
                "detection_class_labels": yolo_data.get("detection_class_labels", []),
                "detection_class_names": yolo_data.get("detection_class_names", []),
                "detection_class_entities": yolo_data.get("detection_class_entities", [])
            }
            
        except Exception as e:
            logger.warning(f"Failed to load YOLO data from {yolo_file}: {str(e)}")
            return {}

    async def _load_yolo_data_with_fallback(
        self, 
        numeric_idx: int, 
        yolo_files: List[str], 
        frame_path: str
    ) -> Dict[str, Any]:
        """Load YOLO data with fallback to real-time detection"""
        # First try to load from preprocessed files
        yolo_data = self._load_yolo_data(numeric_idx, yolo_files)
        
        if yolo_data and yolo_data.get("detected_objects"):
            logger.debug(f"Using preprocessed YOLO data for frame {numeric_idx}")
            return yolo_data
        
        # Fallback to real-time detection
        logger.info(f"No preprocessed YOLO data for frame {numeric_idx}, using fallback detection")
        
        try:
            from app.models.object_detector import object_detector
            
            # Ensure object detector is loaded
            if not object_detector.is_loaded():
                logger.info("Loading object detector for fallback detection...")
                object_detector.load_model()
            
            # Run detection
            detection_result = await object_detector.detect(frame_path)
            
            # Convert to expected format
            detected_objects = [obj.lower() for obj in detection_result.get("detection_class_entities", [])]
            
            fallback_data = {
                "detected_objects": detected_objects,
                "detection_scores": detection_result.get("detection_scores", []),
                "detection_boxes": detection_result.get("detection_boxes", []),
                "detection_class_labels": detection_result.get("detection_class_labels", []),
                "detection_class_names": detection_result.get("detection_class_names", []),
                "detection_class_entities": detection_result.get("detection_class_entities", []),
                "source": "fallback_detection"  # Mark as fallback
            }
            
            logger.info(f"Fallback detection found {len(detected_objects)} objects for frame {numeric_idx}")
            return fallback_data
            
        except Exception as e:
            logger.error(f"Fallback object detection failed for frame {numeric_idx}: {str(e)}")
            return {
                "detected_objects": [],
                "detection_scores": [],
                "detection_boxes": [],
                "detection_class_labels": [],
                "detection_class_names": [],
                "detection_class_entities": [],
                "source": "fallback_failed"
            }

    async def _index_frames(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index frames in vector database and Elasticsearch"""
        logger.info("Indexing frames in databases...")
        
        indexed_count = 0
        failed_count = 0
        
        for frame_result in frame_results:
            try:
                scene_id = frame_result["scene_id"]
                
                # Prepare embeddings for vector database
                embeddings = {}
                visual_data = frame_result.get("visual_data", {})
                
                if "visual_embedding" in visual_data:
                    embeddings["image"] = visual_data["visual_embedding"]
                
                # Prepare metadata for vector database
                scene_metadata = {
                    "video_id": frame_result["video_id"],
                    "start_time": frame_result["start_time"],
                    "end_time": frame_result["end_time"],
                    "duration": frame_result["duration"],
                    "frame_idx": frame_result["frame_idx"],
                    "idx_frame": frame_result["frame_idx"],  # Compatibility
                    "detected_objects": frame_result.get("yolo_data", {}).get("detected_objects", []),
                    "transcript": "",  # No audio for frames
                    "ocr_text": visual_data.get("ocr_text", ""),
                    "scene_description": visual_data.get("scene_label", "")
                }
                
                # Add to vector database
                if embeddings:
                    await vector_db.add_scene_embeddings(scene_id, scene_metadata, embeddings)
                
                # Prepare document for Elasticsearch
                es_doc = {
                    "video_id": frame_result["video_id"],
                    "scene_id": scene_id,
                    "frame_idx": frame_result["frame_idx"],
                    "idx_frame": frame_result["frame_idx"],
                    "start_time": frame_result["start_time"],
                    "transcript": "",
                    "ocr_text": visual_data.get("ocr_text", ""),
                    "detected_objects": scene_metadata["detected_objects"],
                    "scene_description": scene_metadata["scene_description"],
                    "metadata": frame_result.get("yolo_data", {})
                }
                
                # Add to Elasticsearch
                await metadata_db.index_scene(scene_id, es_doc)
                
                indexed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to index frame {frame_result.get('scene_id', 'unknown')}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Indexing completed: {indexed_count} success, {failed_count} failed")
        return {
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "total_count": len(frame_results)
        }

    def _generate_summary(
        self, 
        video_id: str, 
        frame_results: List[Dict[str, Any]], 
        indexing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ingestion summary"""
        return {
            "video_id": video_id,
            "status": "success",
            "total_frames": len(frame_results),
            "processed_frames": len([f for f in frame_results if f.get("visual_data")]),
            "indexed_frames": indexing_results["indexed_count"],
            "failed_indexing": indexing_results["failed_count"],
            "processing_summary": {
                "visual_pipeline": "completed",
                "audio_pipeline": "skipped (frame-level)",
                "vector_database": "indexed",
                "elasticsearch": "indexed"
            }
        }

    async def ingest_multiple_videos(self, video_ids: List[str]) -> Dict[str, Any]:
        """Ingest multiple videos"""
        results = {}
        
        for video_id in video_ids:
            logger.info(f"Processing video {video_id}")
            result = await self.ingest_video(video_id)
            results[video_id] = result
        
        return results


async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Unified video data ingestion")
    parser.add_argument("--video-id", help="Single video ID to process")
    parser.add_argument("--video-ids", nargs="+", help="Multiple video IDs to process")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--all", action="store_true", help="Process all available videos")
    
    args = parser.parse_args()
    
    # Initialize ingestion pipeline
    ingestion = UnifiedIngestion(args.data_dir)
    await ingestion.initialize_services()
    
    # Determine which videos to process
    video_ids = []
    if args.video_id:
        video_ids = [args.video_id]
    elif args.video_ids:
        video_ids = args.video_ids
    elif args.all:
        # Find all available videos
        keyframes_dir = Path(args.data_dir) / "keyframes"
        if keyframes_dir.exists():
            video_ids = [f.stem for f in keyframes_dir.glob("*.csv")]
    
    if not video_ids:
        logger.error("No videos specified. Use --video-id, --video-ids, or --all")
        return
    
    logger.info(f"Processing {len(video_ids)} videos: {video_ids}")
    
    # Process videos
    if len(video_ids) == 1:
        result = await ingestion.ingest_video(video_ids[0])
        logger.info(f"Ingestion result: {result}")
    else:
        results = await ingestion.ingest_multiple_videos(video_ids)
        
        # Print summary
        successful = len([r for r in results.values() if r.get("status") == "success"])
        logger.info(f"Batch ingestion completed: {successful}/{len(video_ids)} successful")
        
        for video_id, result in results.items():
            if result.get("status") == "success":
                logger.info(f"✓ {video_id}: {result['processed_frames']} frames processed")
            else:
                logger.error(f"✗ {video_id}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
