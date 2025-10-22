"""
Ingest API: upload video, extract scene, store in database
"""
import os
import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from loguru import logger

from app.db.database import get_db
from app.db.repository import VideoRepository, ProcessingJobRepository, SceneRepository, FrameRepository
from app.config import settings
from app.services.scene_segmentation import SceneSegmenter
from app.services.visual_pipeline import VisualPipeline
from app.services.audio_pipeline import AudioPipeline
from app.db.vector_db import vector_db
from app.db.metadata_db import metadata_db


router = APIRouter()


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Upload video and trigger scene extraction and processing
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file format
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.SUPPORTED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {settings.SUPPORTED_VIDEO_FORMATS}"
            )
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        filename = f"{video_id}{file_ext}"
        
        # Save file
        upload_path = os.path.join(settings.UPLOAD_DIR, filename)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        with open(upload_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Video uploaded: {filename} ({file_size} bytes)")
        
        # Create video record in database
        video_repo = VideoRepository(db)
        video_data = {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": upload_path,
            "file_size": file_size,
            "format": file_ext,
            "processed": False
        }
        
        video = video_repo.create_video(video_data)
        
        # Create processing job
        job_repo = ProcessingJobRepository(db)
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "video_id": video_id,
            "status": "pending"
        }
        
        job = job_repo.create_job(job_data)
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(process_video_background, video_id, job_id, upload_path)
        
        return JSONResponse(content={
            "status": "uploaded",
            "video_id": video_id,
            "job_id": job_id,
            "message": "Video uploaded successfully. Processing started in background."
        })
        
    except Exception as e:
        logger.error(f"Video upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status/{job_id}")
async def get_processing_status(job_id: str, db: Session = Depends(get_db)):
    """Get processing job status"""
    try:
        job_repo = ProcessingJobRepository(db)
        job = job_repo.get_job_by_id(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JSONResponse(content={
            "job_id": job.job_id,
            "video_id": job.video_id,
            "status": job.status,
            "progress": job.progress,
            "error_message": job.error_message,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        })
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/videos")
async def get_videos(
    limit: int = 20,
    offset: int = 0,
    processed_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get list of videos"""
    try:
        video_repo = VideoRepository(db)
        
        if processed_only:
            videos = video_repo.get_processed_videos()
        else:
            videos = video_repo.get_all_videos(limit=limit, offset=offset)
        
        video_list = []
        for video in videos:
            video_list.append({
                "video_id": video.video_id,
                "filename": video.filename,
                "file_size": video.file_size,
                "duration": video.duration,
                "processed": video.processed,
                "created_at": video.created_at.isoformat() if video.created_at else None
            })
        
        return JSONResponse(content={
            "videos": video_list,
            "total": len(video_list)
        })
        
    except Exception as e:
        logger.error(f"Failed to get videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get videos: {str(e)}")


async def process_video_background(video_id: str, job_id: str, video_path: str):
    """
    Background task to process video: segment scenes, extract features, store in databases
    """
    try:
        logger.info(f"Starting background processing for video: {video_id}")
        
        # Update job status to processing
        from app.db.database import SessionLocal
        db = SessionLocal()
        job_repo = ProcessingJobRepository(db)
        job_repo.update_job_status(job_id, "processing", progress=0.1)
        
        # Step 1: Scene segmentation
        logger.info("Step 1: Scene segmentation")
        segmenter = SceneSegmenter()
        scenes = segmenter.segment(video_path)
        job_repo.update_job_status(job_id, "processing", progress=0.3)
        
        # Step 2: Process each scene
        logger.info(f"Step 2: Processing {len(scenes)} scenes")
        
        # Initialize pipelines
        visual_pipeline = VisualPipeline()  # TODO: Initialize with models
        audio_pipeline = AudioPipeline()    # TODO: Initialize with models
        
        scene_repo = SceneRepository(db)
        frame_repo = FrameRepository(db)
        
        for i, scene_info in enumerate(scenes):
            try:
                # Create scene record
                scene_data = {
                    "scene_id": scene_info["scene_id"],
                    "video_id": video_id,
                    "start_time": scene_info["start_ts"],
                    "end_time": scene_info["end_ts"],
                    "duration": scene_info["end_ts"] - scene_info["start_ts"]
                }
                
                scene = scene_repo.create_scene(scene_data)
                
                # Process scene with visual pipeline
                # TODO: Implement visual pipeline processing
                # frame_data = visual_pipeline.process_scene(video_path, scene_info["scene_id"])
                
                # Process scene with audio pipeline
                # TODO: Implement audio pipeline processing
                # audio_data = audio_pipeline.process_scene_audio(video_path, scene_info["scene_id"], ...)
                
                # Store embeddings in vector database
                # TODO: Store scene embeddings
                
                # Store metadata in Elasticsearch
                # TODO: Store scene metadata
                
                progress = 0.3 + (0.6 * (i + 1) / len(scenes))
                job_repo.update_job_status(job_id, "processing", progress=progress)
                
            except Exception as e:
                logger.error(f"Failed to process scene {scene_info['scene_id']}: {str(e)}")
                continue
        
        # Step 3: Update video as processed
        video_repo = VideoRepository(db)
        video_repo.update_video(video_id, {"processed": True})
        
        # Step 4: Complete job
        job_repo.update_job_status(job_id, "completed", progress=1.0)
        
        logger.info(f"Background processing completed for video: {video_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for video {video_id}: {str(e)}")
        
        # Update job status to failed
        try:
            job_repo.update_job_status(job_id, "failed", error_message=str(e))
        except:
            pass
        
    finally:
        db.close()
