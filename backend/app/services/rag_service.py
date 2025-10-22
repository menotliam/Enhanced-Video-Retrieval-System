import json
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .blip_service import get_blip_service
from .llm_service import LLMService
from ..config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.blip_service = get_blip_service(use_fast=False)  # Use large model
        self.llm_service = LLMService()
        self.caption_cache = {}  # Simple in-memory cache
        self._frame_meta_cache: Dict[str, Dict[str, Any]] = {}
        
    def _load_obj_metadata(self, video_id: str, frame_idx: int) -> List[Dict[str, Any]]:
        """Load object metadata for a specific frame"""
        try:
            # Try to load from obj_index.json first
            obj_index_path = "data/embeddings_v2/obj_index.json"
            if os.path.exists(obj_index_path):
                with open(obj_index_path, 'r', encoding='utf-8') as f:
                    obj_index = json.load(f)
                
                # Find objects for this frame
                frame_objects = []
                for obj_data in obj_index:  # obj_index is a list, not dict
                    if (obj_data.get('video_id') == video_id and 
                        obj_data.get('frame_idx') == frame_idx):
                        frame_objects.append({
                            'doi_tuong': obj_data.get('obj_id', ''),
                            'ten_doi_tuong': obj_data.get('label_vi', '')
                        })
                return frame_objects
            
            # Fallback to meta files
            meta_dir = f"data/meta/obj_meta/{video_id}"
            if os.path.exists(meta_dir):
                frame_objects = []
                meta_file = f"{frame_idx:03d}_obj_meta.json"
                meta_path = os.path.join(meta_dir, meta_file)
                
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        obj_metas = json.load(f)
                    
                    for obj_id, obj_data in obj_metas.items():
                        frame_objects.append({
                            'doi_tuong': obj_id,
                            'ten_doi_tuong': obj_data.get('label_vi', '')
                            
                            
                        })
                return frame_objects
                
        except Exception as e:
            logger.error(f"Error loading object metadata: {e}")
        
        return []
    
    def _get_frame_number_from_idx(self, video_id: str, frame_idx: int) -> int:
        """Get frame number from frame_idx using keyframes CSV"""
        try:
            csv_path = f"data/keyframess/{video_id}.csv"
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                # Find the row where frame_idx matches
                matching_row = df[df['frame_idx'] == frame_idx]
                if not matching_row.empty:
                    return int(matching_row.iloc[0]['n'])
            return frame_idx  # Fallback to frame_idx if not found
        except Exception as e:
            logger.warning(f"Error getting frame number: {e}")
            return frame_idx
    
    def _get_cached_caption(self, image_path: str) -> Optional[str]:
        """Get cached caption if available"""
        return self.caption_cache.get(image_path)
    
    def _cache_caption(self, image_path: str, caption: str):
        """Cache generated caption"""
        self.caption_cache[image_path] = caption
    
    def _generate_frame_caption(self, frame_path: str) -> str:
        """Generate BLIP caption for frame"""
        cache_key = f"frame_{frame_path}"
        cached = self._get_cached_caption(cache_key)
        if cached:
            logger.debug(f"Using cached frame caption for {frame_path}")
            return cached
        
        logger.info(f"Generating BLIP caption for frame: {frame_path}")
        caption = self.blip_service.generate_caption(frame_path, max_length=100)
        
        if caption:
            self._cache_caption(cache_key, caption)
            return caption
        return "Không thể mô tả nội dung hình ảnh"
    
    def _generate_object_captions(self, video_id: str, frame_idx: int) -> List[Dict[str, Any]]:
        """Generate BLIP captions for objects in frame"""
        obj_captions = []
        
        # Get object metadata
        objects = self._load_obj_metadata(video_id, frame_idx)
        
        for obj in objects:
            obj_id = obj['doi_tuong']
            # Get frame number from frame_idx
            frame_number = self._get_frame_number_from_idx(video_id, frame_idx)
            obj_path = f"data/obj_det/{video_id}/{frame_number:03d}_{obj_id}.jpg"
            
            if os.path.exists(obj_path):
                cache_key = f"obj_{obj_path}"
                cached = self._get_cached_caption(cache_key)
                
                if cached:
                    logger.debug(f"Using cached object caption for {obj_path}")
                    caption = cached
                else:
                    logger.info(f"Generating BLIP caption for object: {obj_path}")
                    caption = self.blip_service.generate_caption(obj_path, max_length=50)
                    if caption:
                        self._cache_caption(cache_key, caption)
                    else:
                        caption = "Không thể mô tả đối tượng"
                
                obj_captions.append({
                    'doi_tuong': obj_id,
                    'ten_doi_tuong': obj['ten_doi_tuong'],
                    'mo_ta': caption
                    
                })
        
        return obj_captions

    def _load_frame_metadata(self, video_id: str, frame_idx: int) -> Dict[str, Any]:
        """Load per-frame metadata (asr_text, ocr_texts, scene_label, etc.).

        Priority order:
        1) data/embeddings_v2/meta/frame_meta/{video_id}/{frame_idx}.json (fast, small)
        2) Fallback: scan data/embeddings_v2/frame_index.json for matching record (slow for very large files)
        """
        cache_key = f"{video_id}:{frame_idx}"
        if cache_key in self._frame_meta_cache:
            return self._frame_meta_cache[cache_key]

        # 1) Try per-frame meta JSON
        try:
            meta_path = Path("data/embeddings_v2/meta/frame_meta") / video_id / f"{frame_idx}.json"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Normalize fields
                meta = {
                    'video_id': data.get('video_id', video_id),
                    'frame_idx': int(data.get('frame_idx', frame_idx)),
                    'asr_text': data.get('asr_text') or data.get('asr') or '',
                    'ocr_texts': data.get('ocr_texts') or data.get('ocr') or [],
                    'scene_label': data.get('scene_label') or data.get('scene') or ''
                }
                self._frame_meta_cache[cache_key] = meta
                return meta
        except Exception as e:
            logger.warning(f"Failed reading frame meta json for {cache_key}: {e}")

        # 2) Fallback: scan frame_index.json (may be large)
        try:
            index_path = Path("data/embeddings_v2/frame_index.json")
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                for it in items:
                    if it.get('video_id') == video_id and int(it.get('frame_idx', -1)) == int(frame_idx):
                        meta = {
                            'video_id': it.get('video_id', video_id),
                            'frame_idx': int(it.get('frame_idx', frame_idx)),
                            'asr_text': it.get('asr_text', ''),
                            'ocr_texts': it.get('ocr_texts', []),
                            'scene_label': it.get('scene_label', '')
                        }
                        self._frame_meta_cache[cache_key] = meta
                        return meta
        except Exception as e:
            logger.warning(f"Failed scanning frame_index.json for {cache_key}: {e}")

        # Default empty
        meta = {
            'video_id': video_id,
            'frame_idx': int(frame_idx),
            'asr_text': '',
            'ocr_texts': [],
            'scene_label': ''
        }
        self._frame_meta_cache[cache_key] = meta
        return meta
    
    def _build_context(self, frame_data: Dict[str, Any], frame_caption: str, obj_captions: List[Dict[str, Any]]) -> str:
        """Build context string for LLM"""
        context_parts = []
        
        # Frame caption
        context_parts.append(f"Caption khung hình: {frame_caption}")
        
        # Object captions
        if obj_captions:
            obj_parts = []
            for obj in obj_captions:
                obj_part = f"Đối tượng {obj['doi_tuong']} ({obj['ten_doi_tuong']}): {obj['mo_ta']}"
                obj_parts.append(obj_part)
            context_parts.append("Caption các đối tượng của frame:\n" + "\n".join(obj_parts))
        
        # ASR text
        if frame_data.get('asr_text'):
            context_parts.append(f"Lời nói (ASR): {frame_data['asr_text']}")
        
        # OCR texts
        if frame_data.get('ocr_texts'):
            ocr_text = " ".join(frame_data['ocr_texts']) if isinstance(frame_data['ocr_texts'], list) else frame_data['ocr_texts']
            context_parts.append(f"Văn bản trong hình (OCR): {ocr_text}")
        
        # Scene label
        # if frame_data.get('scene_label'):
        #     context_parts.append(f"Nhãn cảnh: {frame_data['scene_label']}")
        
        return "\n\n".join(context_parts)
    
    async def generate_rag_answer(self, top_frame: Any, query: str) -> Optional[str]:
        """
        Generate RAG answer using the complete pipeline
        
        Args:
            top_frame: Top retrieved frame data
            query: User query
            
        Returns:
            Generated answer or None if error
        """
        try:
            logger.info("Starting RAG pipeline...")
            
            # Accept either dict with fields or a string "<video_id>:<frame_idx>"
            if isinstance(top_frame, str):
                try:
                    video_id, frame_idx_str = top_frame.split(":", 1)
                    frame_idx = int(frame_idx_str)
                except Exception:
                    logger.error(f"Invalid top_frame format: {top_frame}")
                    return None
                base_frame = {'video_id': video_id, 'frame_idx': frame_idx}
            elif isinstance(top_frame, dict):
                video_id = top_frame.get('video_id')
                frame_idx = int(top_frame.get('frame_idx'))
                base_frame = {'video_id': video_id, 'frame_idx': frame_idx}
            else:
                logger.error(f"Unsupported top_frame type: {type(top_frame)}")
                return None

            # Auto-load metadata (asr_text, ocr_texts, scene_label)
            meta = self._load_frame_metadata(video_id, frame_idx)
            enriched_frame = {**base_frame, **meta}
            # Get frame number from frame_idx
            frame_number = self._get_frame_number_from_idx(video_id, int(frame_idx))
            frame_path = f"data/keyframes/{video_id}/{frame_number:03d}.jpg"
            print(f"Frame path: {frame_path}")
            logger.info(f"Processing frame: {video_id}/{frame_idx}")
            
            # Step 1: Generate frame caption
            if os.path.exists(frame_path):
                frame_caption = self._generate_frame_caption(frame_path)
                logger.info(f"Frame caption: {frame_caption}")
            else:
                logger.warning(f"Frame not found: {frame_path}")
                frame_caption = "Không tìm thấy hình ảnh"
            
            # Step 2: Generate object captions
            obj_captions = self._generate_object_captions(video_id, int(frame_idx))
            logger.info(f"Generated {len(obj_captions)} object captions")
            
            # Step 3: Build context
            context = self._build_context(enriched_frame, frame_caption, obj_captions)
            logger.info(f"Built context length: {len(context)} chars")
            print(f"Context: {context}")
            # Step 4: Generate answer with LLM
            answer = await self.llm_service.generate_answer_with_context(query, context)
            
            logger.info("RAG pipeline completed successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return None

# Global instance
rag_service = None

def get_rag_service() -> RAGService:
    """Get or create RAG service instance"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service
