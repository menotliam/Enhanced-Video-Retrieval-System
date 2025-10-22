"""
Search API endpoints for the AI Video Search system
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Dict, Any, List
import time
from loguru import logger
import sys
from pathlib import Path
import importlib.util
import csv
import io
import urllib.parse
import json

#from app.services.search_service import search_service
#from app.utils.text_normalizer import text_normalizer
from app.config import settings
from app.services.llm_service import LLMService
from app.services.rag_service import get_rag_service

router = APIRouter()

# Cache for media info to avoid repeated file reads
_media_info_cache = {}
_keyframes_cache = {}

def load_media_info(video_id: str, data_root: str) -> Optional[Dict[str, Any]]:
    """Load media info for a video_id from the media_info folder"""
    if video_id in _media_info_cache:
        return _media_info_cache[video_id]
    
    try:
        media_info_path = Path(data_root) / "media_info" / f"{video_id}.json"
        if media_info_path.exists():
            with open(media_info_path, 'r', encoding='utf-8') as f:
                media_info = json.load(f)
                _media_info_cache[video_id] = media_info
                return media_info
    except Exception as e:
        logger.warning(f"Failed to load media info for {video_id}: {e}")
    
    _media_info_cache[video_id] = None
    return None

def load_keyframes_data(video_id: str, data_root: str) -> Optional[Dict[int, float]]:
    """Load keyframes data for a video_id from the keyframes folder.
    Returns a mapping of frame_idx to pts_time (timestamp in seconds).
    """
    if video_id in _keyframes_cache:
        return _keyframes_cache[video_id]
    
    try:
        keyframes_path = Path(data_root) / "keyframes" / f"{video_id}.csv"
        if keyframes_path.exists():
            frame_to_time = {}
            with open(keyframes_path, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        try:
                            frame_idx = int(parts[3])  # frame_idx is the 4th column
                            pts_time = float(parts[1])  # pts_time is the 2nd column
                            frame_to_time[frame_idx] = pts_time
                        except (ValueError, IndexError):
                            continue
            _keyframes_cache[video_id] = frame_to_time
            return frame_to_time
    except Exception as e:
        logger.warning(f"Failed to load keyframes data for {video_id}: {e}")
    
    _keyframes_cache[video_id] = None
    return None


# @router.post("/text")
# async def search_by_text(
#     query: str = Form(..., description="Vietnamese text query"),
#     top_k: int = Query(20, description="Number of results to return"),
#     use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
#     filters: Optional[str] = Form(None, description="JSON string of additional filters")
# ):
#     """
#     Search videos by Vietnamese text query
#     """
#     try:
#         start_time = time.time()
        
#         # Parse filters if provided
#         parsed_filters = None
#         if filters:
#             import json
#             try:
#                 parsed_filters = json.loads(filters)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
#         # Normalize query
#         normalized_query = text_normalizer.normalize_text(query)
        
#         logger.info(f"Text search request: {query} -> {normalized_query}")
        
#         # Perform search
#         results = await search_service.search(
#             query=normalized_query,
#             query_type="text",
#             filters=parsed_filters,
#             top_k=top_k,
#             use_rag=use_rag
#         )
        
#         # Add search time and query
#         search_time = time.time() - start_time
#         results["search_time"] = search_time
#         results["query"] = query
        
#         logger.info(f"Text search completed in {search_time:.2f}s, found {results['total_results']} results")
        
#         return JSONResponse(content=results)
        
#     except Exception as e:
#         logger.error(f"Text search failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @router.post("/image")
# async def search_by_image(
#     image: UploadFile = File(..., description="Image file for similarity search"),
#     query: Optional[str] = Form(None, description="Additional text query"),
#     top_k: int = Query(20, description="Number of results to return"),
#     use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
#     filters: Optional[str] = Form(None, description="JSON string of additional filters")
# ):
#     """
#     Search videos by image similarity
#     """
#     try:
#         start_time = time.time()
        
#         # Validate image file
#         if not image.content_type.startswith("image/"):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read image bytes
#         image_bytes = await image.read()
        
#         # Parse filters if provided
#         parsed_filters = None
#         if filters:
#             import json
#             try:
#                 parsed_filters = json.loads(filters)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
#         logger.info(f"Image search request: {image.filename}, size: {len(image_bytes)} bytes")
        
#         # Perform search
#         results = await search_service.search(
#             query=query or "",
#             query_type="image",
#             query_image=image_bytes,
#             filters=parsed_filters,
#             top_k=top_k,
#             use_rag=use_rag
#         )
        
#         # Add search time and query
#         search_time = time.time() - start_time
#         results["search_time"] = search_time
#         results["query"] = query or f"Image search: {image.filename}"
        
#         logger.info(f"Image search completed in {search_time:.2f}s, found {results['total_results']} results")
        
#         return JSONResponse(content=results)
        
#     except Exception as e:
#         logger.error(f"Image search failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @router.post("/audio")
# async def search_by_audio(
#     audio: UploadFile = File(..., description="Audio file for speech search"),
#     query: Optional[str] = Form(None, description="Additional text query"),
#     top_k: int = Query(20, description="Number of results to return"),
#     use_rag: bool = Query(False, description="Whether to use RAG for answer generation"),
#     filters: Optional[str] = Form(None, description="JSON string of additional filters")
# ):
#     """
#     Search videos by audio/speech content
#     """
#     try:
#         start_time = time.time()
        
#         # Validate audio file
#         if not audio.content_type.startswith("audio/"):
#             raise HTTPException(status_code=400, detail="File must be an audio file")
        
#         # Read audio bytes
#         audio_bytes = await audio.read()
        
#         # Parse filters if provided
#         parsed_filters = None
#         if filters:
#             import json
#             try:
#                 parsed_filters = json.loads(filters)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
#         logger.info(f"Audio search request: {audio.filename}, size: {len(audio_bytes)} bytes")
        
#         # Perform search
#         results = await search_service.search(
#             query=query or "",
#             query_type="audio",
#             query_audio=audio_bytes,
#             filters=parsed_filters,
#             top_k=top_k,
#             use_rag=use_rag
#         )
        
#         # Add search time and query
#         search_time = time.time() - start_time
#         results["search_time"] = search_time
#         results["query"] = query or f"Audio search: {audio.filename}"
        
#         logger.info(f"Audio search completed in {search_time:.2f}s, found {results['total_results']} results")
        
#         return JSONResponse(content=results)
        
#     except Exception as e:
#         logger.error(f"Audio search failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# @router.get("/suggest")
# async def get_search_suggestions(
#     query: str = Query(..., description="Partial query for suggestions"),
#     limit: int = Query(10, description="Number of suggestions to return")
# ):
#     """
#     Get search suggestions based on partial query
#     """
#     try:
#         # Normalize query
#         normalized_query = text_normalizer.normalize_text(query)
        
#         # Extract keywords
#         keywords = text_normalizer.extract_keywords(normalized_query)
        
#         # Generate suggestions (simple approach)
#         suggestions = []
        
#         # Add keyword-based suggestions
#         for keyword in keywords[:limit//2]:
#             suggestions.append(f"Tìm kiếm {keyword}")
#             suggestions.append(f"Video có {keyword}")
        
#         # Add common Vietnamese search patterns
#         common_patterns = [
#             "người",
#             "xe",
#             "nhà",
#             "đường",
#             "cửa hàng",
#             "công viên",
#             "trường học",
#             "bệnh viện"
#         ]
        
#         for pattern in common_patterns:
#             if pattern not in suggestions and len(suggestions) < limit:
#                 suggestions.append(pattern)
        
#         return JSONResponse(content={
#             "query": query,
#             "suggestions": suggestions[:limit]
#         })
        
#     except Exception as e:
#         logger.error(f"Search suggestions failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")


# @router.get("/filters")
# async def get_available_filters():
#     """
#     Get available filters for search
#     """
#     try:
#         filters = {
#             "objects": [
#                 "người", "xe", "cây", "nhà", "đường", "cửa hàng",
#                 "công viên", "trường học", "bệnh viện", "xe máy",
#                 "ô tô", "xe đạp", "mèo", "chó", "chim"
#             ],
#             "locations": [
#                 "hà nội", "sài gòn", "hồ chí minh", "tp.hcm", "hcm",
#                 "đà nẵng", "huế", "nha trang", "vũng tàu"
#             ],
#             "time_of_day": [
#                 "sáng", "trưa", "chiều", "tối", "đêm"
#             ],
#             "weather": [
#                 "nắng", "mưa", "mây", "trời quang"
#             ]
#         }
        
#         return JSONResponse(content=filters)
        
#     except Exception as e:
#         logger.error(f"Get filters failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get filters: {str(e)}")


# @router.get("/statistics")
# async def get_search_statistics():
#     """
#     Get search system statistics
#     """
#     try:
#         # Get vector database statistics
#         vector_stats = search_service.vector_db.get_statistics()
        
#         # Get metadata database statistics
#         metadata_stats = await search_service.metadata_db.get_statistics()
        
#         # Get embedding model info
#         embedding_info = {
#             "phobert_loaded": "phobert" in search_service.embedding_model.models,
#             "clip_loaded": "clip" in search_service.embedding_model.models,
#             "asr_loaded": search_service.asr_model.model is not None,
#             "cross_encoder_loaded": search_service.cross_encoder.model is not None
#         }
        
#         # Get LLM service info
#         llm_info = {
#             "openai_available": LLMService.is_available(),
#             "offline_mode": not settings.USE_LLM
#         }
        
#         statistics = {
#             "vector_database": vector_stats,
#             "metadata_database": metadata_stats,
#             "models": embedding_info,
#             "llm_service": llm_info,
#             "system_info": {
#                 "version": settings.VERSION,
#                 "vector_dimension": settings.VECTOR_DIMENSION,
#                 "fusion_enabled": True
#             }
#         }
        
#         return JSONResponse(content=statistics)
        
#     except Exception as e:
#         logger.error(f"Get statistics failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# @router.post("/health")
# async def health_check():
#     """
#     Health check for search service
#     """
#     try:
#         # Check if all components are available
#         health_status = {
#             "status": "healthy",
#             "components": {
#                 "embedding_models": "phobert" in search_service.embedding_model.models,
#                 "asr_model": search_service.asr_model.model is not None,
#                 "cross_encoder": search_service.cross_encoder.model is not None,
#                 "vector_db": search_service.vector_db.indices["text"] is not None,
#                 "metadata_db": search_service.metadata_db.es_client is not None,
#                 "llm_service": LLMService.is_available(),
#                 "fusion_service": True  # Always available
#             }
#         }
        
#         # Check if all components are healthy
#         all_healthy = all(health_status["components"].values())
#         health_status["status"] = "healthy" if all_healthy else "degraded"
        
#         return JSONResponse(content=health_status)
        
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         return JSONResponse(content={
#             "status": "unhealthy",
#             "error": str(e)
#         }, status_code=500)


@router.post("/frames-cli")
async def search_frames_cli(
    query: str = Form(..., description="Text query for frame search"),
    top_k: int = Query(10, description="Number of results to return"),
    pre_filter: bool = Query(False, description="Enable pre-filter using metadata"),
    pre_filter_limit: Optional[int] = Query(None, description="Cap number of pre-filter candidates"),
):
    """Wrap the CLI logic to search frames and return frame_idx with image_url.

    Response format:
    {
      "query": str,
      "total_results": int,
      "results": [
        {"video_id": str, "frame_idx": int, "score": float, "image_url": str, "image": str, "youtube_url": str, "frame_timestamp": float}
      ]
    }
    """
    try:
        start_time = time.time()

        # Dynamically import the CLI module function by file path
        backend_root = Path(__file__).resolve().parents[2]
        cli_path = backend_root / "scripts" / "search_text_frame.py"
        if not cli_path.exists():
            raise HTTPException(status_code=500, detail="CLI script not found")

        spec = importlib.util.spec_from_file_location("search_text_frame", str(cli_path))
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail="Failed to load CLI script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["search_text_frame"] = module
        spec.loader.exec_module(module)  # type: ignore

        if not hasattr(module, "search_frames_programmatic"):
            raise HTTPException(status_code=500, detail="CLI function not available")

        data_root = str(backend_root / "data")

        # Execute search
        result = await module.search_frames_programmatic(
            query=query,
            data_root=data_root,
            embeddings_root=str(Path(data_root) / "embeddings_v2"),
            index_dir=str(Path(data_root) / "faiss_index_v2"),
            top_k=top_k,
            use_gpu=False,
            use_llm=False,
            pre_filter=pre_filter,
            pre_filter_limit=int(pre_filter_limit) if pre_filter_limit else 0,
        )

        raw_items: List[Dict[str, Any]] = result.get("results", [])  # type: ignore

        # Build image_url based on StaticFiles mount: /static -> backend/data
        def to_image_url(image_path: Optional[str]) -> Optional[str]:
            if not image_path:
                return None
            p = Path(image_path)
            # If path contains 'data', strip up to it
            try:
                parts = p.parts
                if "data" in parts:
                    idx = parts.index("data")
                    rel = Path(*parts[idx+1:])
                else:
                    # If already relative to data/, keep as-is
                    rel = p
                return f"/static/{rel.as_posix()}"
            except Exception:
                return None

        formatted: List[Dict[str, Any]] = []
        for item in raw_items:
            img_path = item.get("image_path")
            img_name = None
            try:
                if img_path:
                    img_name = Path(img_path).name
            except Exception:
                img_name = None
            
            # Load media info to get YouTube URL and keyframes data for timestamp
            video_id = item.get("video_id")
            frame_idx = int(item.get("frame_idx", 0))
            youtube_url = None
            frame_timestamp = None
            
            if video_id:
                # Load media info for base YouTube URL
                media_info = load_media_info(video_id, data_root)
                if media_info:
                    base_youtube_url = media_info.get("watch_url")
                    
                    # Load keyframes data for timestamp
                    keyframes_data = load_keyframes_data(video_id, data_root)
                    if keyframes_data and frame_idx in keyframes_data:
                        frame_timestamp = keyframes_data[frame_idx]
                        
                        # Construct YouTube URL with timestamp
                        if base_youtube_url and frame_timestamp is not None:
                            # Add timestamp parameter to YouTube URL
                            if '?' in base_youtube_url:
                                youtube_url = f"{base_youtube_url}&t={int(frame_timestamp)}s"
                            else:
                                youtube_url = f"{base_youtube_url}?t={int(frame_timestamp)}s"
                        else:
                            youtube_url = base_youtube_url
                    else:
                        youtube_url = base_youtube_url
            
            formatted.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "score": float(item.get("score") or 0.0),
                "image_url": to_image_url(img_path),
                "image": img_name,
                "youtube_url": youtube_url,
                "frame_timestamp": frame_timestamp,
            })

        resp = {
            "query": query,
            "total_results": len(formatted),
            "results": formatted,
            "search_time": time.time() - start_time,
        }
        return JSONResponse(content=resp)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frames CLI search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frames CLI search failed: {str(e)}")


@router.get("/frames-cli")
async def search_frames_cli_get(
    query: str = Query(..., description="Text query for frame search"),
    top_k: Optional[int] = Query(None, description="Number of results to return"),
    limit: Optional[int] = Query(None, description="Alias for top_k for compatibility"),
    pre_filter: bool = Query(False, description="Enable pre-filter using metadata"),
    pre_filter_limit: Optional[int] = Query(None, description="Cap number of pre-filter candidates"),
):
    """GET-compatible wrapper for frames-cli. Accepts query params instead of form data.
    Supports both top_k and limit (alias) to avoid 405 from accidental GET calls.
    """
    resolved_top_k = top_k if top_k is not None else (limit if limit is not None else 10)
    # Reuse the POST handler's logic by calling it directly
    # Build a form-like call
    return await search_frames_cli(
        query=query,
        top_k=resolved_top_k,
        pre_filter=pre_filter,
        pre_filter_limit=pre_filter_limit,
    )


@router.post("/rag/generate")
async def generate_rag_answer(
    frame_input: str = Form(..., description="Frame input in format <video_id>:<frame_idx>"),
    query: str = Form(..., description="Question to ask about the frame")
):
    """Generate RAG answer for a specific frame.
    
    Args:
        frame_input: Frame identifier in format "L21_V001:90"
        query: Question about the frame
        
    Returns:
        {
            "answer": str,
            "frame_info": {
                "video_id": str,
                "frame_idx": int,
                "image_url": str
            }
        }
    """
    try:
        start_time = time.time()
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Generate answer
        answer = await rag_service.generate_rag_answer(frame_input, query)
        
        if answer is None:
            raise HTTPException(status_code=500, detail="RAG service error !")
        
        # Parse frame input to get video_id and frame_idx
        try:
            video_id, frame_idx_str = frame_input.split(":", 1)
            frame_idx = int(frame_idx_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid frame format. Use <video_id>:<frame_idx>")
        
        # Get frame number and build image path
        frame_number = rag_service._get_frame_number_from_idx(video_id, frame_idx)
        frame_path = f"data/keyframes/{video_id}/{frame_number:03d}.jpg"
        
        # Check if frame exists
        if not Path(frame_path).exists():
            raise HTTPException(status_code=404, detail="Hình ảnh không tồn tại.")
        
        # Build image URL
        image_url = f"/static/keyframes/{video_id}/{frame_number:03d}.jpg"
        
        response = {
            "answer": answer,
            "frame_info": {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "image_url": image_url
            },
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"RAG answer generated for {frame_input} in {response['processing_time']:.2f}s")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG service error !")


@router.post("/frames-cli/export-csv")
async def search_frames_export_csv(
    query: str = Form(..., description="Text query for frame search"),
    top_k: int = Form(10, description="Number of results to return"),
    pre_filter: bool = Form(False, description="Enable pre-filter using metadata"),
    pre_filter_limit: Optional[int] = Form(None, description="Cap number of pre-filter candidates"),
    filename: Optional[str] = Form(None, description="Custom filename for CSV export")
):
    """Search frames and export results to CSV format.
    
    Returns CSV file with columns: video_id, frame_idx
    """
    try:
        start_time = time.time()
        
        # Debug log
        logger.info(f"CSV Export - Query: {query}, top_k: {top_k}, pre_filter: {pre_filter}")

        # Dynamically import the CLI module function by file path
        backend_root = Path(__file__).resolve().parents[2]
        cli_path = backend_root / "scripts" / "search_text_frame.py"
        if not cli_path.exists():
            raise HTTPException(status_code=500, detail="CLI script not found")

        spec = importlib.util.spec_from_file_location("search_text_frame", str(cli_path))
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail="Failed to load CLI script")
        module = importlib.util.module_from_spec(spec)
        sys.modules["search_text_frame"] = module
        spec.loader.exec_module(module)  # type: ignore

        if not hasattr(module, "search_frames_programmatic"):
            raise HTTPException(status_code=500, detail="CLI function not available")

        data_root = str(backend_root / "data")

        # Execute search
        result = await module.search_frames_programmatic(
            query=query,
            data_root=data_root,
            embeddings_root=str(Path(data_root) / "embeddings_v2"),
            index_dir=str(Path(data_root) / "faiss_index_v2"),
            top_k=int(top_k),  # Ensure it's an integer
            use_gpu=False,
            use_llm=True,
            pre_filter=pre_filter,
            pre_filter_limit=int(pre_filter_limit) if pre_filter_limit else 0,
        )

        raw_items: List[Dict[str, Any]] = result.get("results", [])  # type: ignore

        # Create CSV content with UTF-8 encoding
        output = io.StringIO()
        writer = csv.writer(output)
        
        
        
        # Write data rows
        for item in raw_items:
            video_id = item.get("video_id", "")
            frame_idx = item.get("frame_idx", "")
            writer.writerow([video_id, frame_idx])

        # Get CSV content and encode to UTF-8
        csv_content = output.getvalue()
        output.close()
        
        # Ensure UTF-8 encoding
        if isinstance(csv_content, str):
            csv_content = csv_content.encode('utf-8')

        # Generate filename if not provided
        if not filename:
            timestamp = int(time.time())
            filename = f"search_results_{timestamp}.csv"

        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # URL encode filename for proper handling of Vietnamese characters
        encoded_filename = urllib.parse.quote(filename.encode('utf-8'))

        # Create streaming response
        def generate():
            if isinstance(csv_content, bytes):
                yield csv_content
            else:
                yield csv_content.encode('utf-8')

        return StreamingResponse(
            generate(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


@router.get("/frames-cli/export-csv")
async def search_frames_export_csv_get(
    query: str = Query(..., description="Text query for frame search"),
    top_k: Optional[int] = Query(10, description="Number of results to return"),
    limit: Optional[int] = Query(None, description="Alias for top_k for compatibility"),
    pre_filter: bool = Query(False, description="Enable pre-filter using metadata"),
    pre_filter_limit: Optional[int] = Query(None, description="Cap number of pre-filter candidates"),
    filename: Optional[str] = Query(None, description="Custom filename for CSV export")
):
    """GET-compatible wrapper for CSV export. Accepts query params instead of form data."""
    resolved_top_k = top_k if top_k is not None else (limit if limit is not None else 10)
    # Reuse the POST handler's logic by calling it directly
    return await search_frames_export_csv(
        query=query,
        top_k=resolved_top_k,
        pre_filter=pre_filter,
        pre_filter_limit=pre_filter_limit,
        filename=filename
    )
