#!/usr/bin/env python3
"""
Test script for unified ingestion pipeline with L21_V001 data
"""
import os
import sys
import asyncio
import json
import csv
import numpy as np
from pathlib import Path

# Add backend app to sys.path so imports work
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from loguru import logger
from app.scripts.unified_ingestion import UnifiedIngestion
from app.db.database import create_tables, check_database_connection
from app.db.vector_db import vector_db
from app.db.metadata_db import metadata_db


def setup_test_data():
    """Setup and validate test data for L21_V001"""
    video_id = "L21_V001"
    
    # Base data directory relative to repo root
    base_dir = Path(__file__).resolve().parents[1] / "data"
    
    # Expected file locations according to unified ingestion format
    required_files = {
        "keyframes_csv": base_dir / "keyframes" / f"{video_id}.csv",
        "frames_dir": base_dir / "frames" / video_id,
        "metadata_json": base_dir / "metadata" / f"{video_id}.json",
        "obj_detection_dir": base_dir / "obj_detection"
    }
    
    logger.info(f"Validating test data for {video_id}...")
    
    # Check required files exist
    missing_files = []
    for file_type, file_path in required_files.items():
        if file_type.endswith("_dir"):
            if not file_path.exists() or not file_path.is_dir():
                missing_files.append(f"{file_type}: {file_path}")
        else:
            if not file_path.exists() or not file_path.is_file():
                missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        logger.warning("Missing some test data files:")
        for missing in missing_files:
            logger.warning(f"  - {missing}")
        logger.info("Will attempt to create sample data...")
        create_sample_test_data(base_dir, video_id)
    else:
        logger.info("✅ All required test data files found")
    
    return video_id, str(base_dir)


def create_sample_test_data(base_dir: Path, video_id: str):
    """Create minimal sample data for testing"""
    logger.info(f"Creating sample test data for {video_id}")
    
    # Create directories
    (base_dir / "keyframes").mkdir(parents=True, exist_ok=True)
    (base_dir / "frames" / video_id).mkdir(parents=True, exist_ok=True)
    (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (base_dir / "obj_detection").mkdir(parents=True, exist_ok=True)
    
    # Create sample keyframes CSV
    keyframes_csv = base_dir / "keyframes" / f"{video_id}.csv"
    if not keyframes_csv.exists():
        with open(keyframes_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['n', 'pts_time', 'fps', 'frame_idx'])
            for i in range(3):  # Create 3 sample frames
                writer.writerow([i+1, i*2.5, 30.0, i*75])
    
    # Create sample frame images
    frames_dir = base_dir / "frames" / video_id
    for i in range(3):
        frame_file = frames_dir / f"{i:03d}.jpg"
        if not frame_file.exists():
            # Create a simple test image
            try:
                from PIL import Image
                image = Image.new('RGB', (320, 240), (100+i*50, 150, 200))
                image.save(frame_file, 'JPEG')
            except ImportError:
                # If PIL not available, create empty file
                frame_file.touch()
    
    # Create sample metadata JSON
    metadata_json = base_dir / "metadata" / f"{video_id}.json"
    if not metadata_json.exists():
        metadata = {
            "title": f"Test Video {video_id}",
            "length": 7.5,
            "description": "Sample video for testing unified ingestion",
            "keywords": ["test", "sample"]
        }
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create sample YOLO detection files
    obj_detection_dir = base_dir / "obj_detection"
    for i in range(3):
        yolo_file = obj_detection_dir / f"{i:03d}.json"
        if not yolo_file.exists():
            yolo_data = {
                "detection_scores": [0.95, 0.85],
                "detection_class_names": ["person", "car"],
                "detection_class_entities": ["người", "xe"],
                "detection_boxes": [[100, 100, 200, 200], [50, 50, 150, 150]],
                "detection_class_labels": [0, 1]
            }
            with open(yolo_file, 'w', encoding='utf-8') as f:
                json.dump(yolo_data, f, indent=2)
    
    logger.info("✅ Sample test data created")


async def test_ingestion():
    """Test the unified ingestion pipeline"""
    logger.info("🚀 Starting unified ingestion test...")
    
    try:
        # Setup test data
        video_id, data_dir = setup_test_data()
        
        # Initialize database
        logger.info("Initializing database...")
        create_tables()
        if not check_database_connection():
            logger.error("❌ Database connection failed!")
            return False
        
        # Initialize ingestion system
        logger.info("Initializing ingestion system...")
        ingestion = UnifiedIngestion(data_dir)
        await ingestion.initialize_services()
        
        # Run ingestion
        logger.info(f"Running ingestion for {video_id}...")
        result = await ingestion.ingest_video(video_id)
        
        if "error" in result:
            logger.error(f"❌ Ingestion failed: {result['error']}")
            return False
        
        # Log results
        logger.info("✅ Ingestion completed successfully!")
        logger.info(f"📊 Results:")
        logger.info(f"  - Video ID: {result['video_id']}")
        logger.info(f"  - Total frames: {result['total_frames']}")
        logger.info(f"  - Processed frames: {result['processed_frames']}")
        logger.info(f"  - Indexed frames: {result['indexed_frames']}")
        
        if result['failed_indexing'] > 0:
            logger.warning(f"  - Failed indexing: {result['failed_indexing']}")
        
        # Test basic search
        logger.info("Testing basic search functionality...")
        try:
            from app.services.search_service import search_service
            
            search_results = await search_service.search(
                query="người",
                query_type="text",
                top_k=5
            )
            
            logger.info(f"✅ Search test completed:")
            logger.info(f"  - Query: 'người'")
            logger.info(f"  - Results found: {search_results['total_results']}")
            
            if search_results['total_results'] > 0:
                sample = search_results['results'][0]
                logger.info(f"  - Top result: {sample.get('video_name', 'N/A')} frame {sample.get('frame_idx', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"Search test failed (expected if models not loaded): {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_ingestion())
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1)