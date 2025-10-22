"""
Online Search Pipeline Service
Implements the complete search pipeline: parse query, filter, embed, ANN, re-rank, RAG
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from app.config import settings, MODEL_CONFIG
from app.models.embedding_model import EmbeddingModel
from app.models.asr_model import ASRModel
from app.models.cross_encoder import cross_encoder
from app.db.vector_db import VectorDB
from app.db.metadata_db import MetadataDB
from app.utils.text_normalizer import text_normalizer
from app.services.fusion import FusionService
from app.services.llm_service import llm_service


class SearchService:
    """Main search service implementing the online search pipeline"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.asr_model = ASRModel()
        self.vector_db = VectorDB()
        self.metadata_db = MetadataDB()
        self.fusion_service = FusionService()
        self.cross_encoder = cross_encoder
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all required models"""
        logger.info("Initializing search service models...")
        
        # Initialize embedding models
        self.embedding_model.load_models()
        
        # Initialize ASR model
        self.asr_model.load_model()
        
        # Initialize cross-encoder model
        self.cross_encoder.load_model()
        
        # Initialize databases
        self.vector_db.initialize()
        self.metadata_db.initialize()
        
        logger.info("Search service models initialized successfully")
    
    async def search(
        self, 
        query: str,
        query_type: str = "text",  # "text", "image", "audio"
        query_image: Optional[bytes] = None,
        query_audio: Optional[bytes] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        use_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Main search method implementing the complete pipeline
        
        Args:
            query: Text query (Vietnamese)
            query_type: Type of query ("text", "image", "audio")
            query_image: Image bytes for image query
            query_audio: Audio bytes for audio query
            filters: Additional filters
            top_k: Number of results to return
            use_rag: Whether to use RAG for answer generation
            
        Returns:
            Search results with video names, frame indices, and optional RAG answer
        """
        try:
            logger.info(f"Starting search pipeline for query: {query[:50]}...")
            
            # Step 1: Query parsing and expansion using LLM
            parsed_query = await self._parse_and_expand_query(query, filters)
            
            # Step 2: Handle different query types
            if query_type == "audio":
                query = await self._process_audio_query(query_audio)
            elif query_type == "image":
                # Image query processing is handled in encoding step
                pass
            elif query_type == "text":
                # Text query processing is handled in encoding step
                pass
            
            # Step 3: Pre-filter metadata
            filtered_scene_ids = await self._pre_filter_metadata(parsed_query)
            
            # Step 4: Encode query to embeddings
            query_embeddings = await self._encode_query(
                parsed_query, query_type, query_image, query_audio
            )
            
            # Step 5: Fusion search (primary) with fallback to separate modalities
            candidate_scenes = await self._fusion_search(
                query_embeddings, query_type, filtered_scene_ids, top_k * 2
            )
            
            # Step 6: Enrich candidates with metadata for re-ranking
            enriched_scenes = await self._enrich_scenes_with_metadata(candidate_scenes)
            
            # Step 7: Re-ranking with Vietnamese cross-encoder
            ranked_scenes = await self._re_rank_scenes(
                enriched_scenes, parsed_query, top_k
            )
            
            # Step 8: Optional RAG
            rag_answer = None
            if use_rag:
                rag_answer = await self._generate_rag_answer(ranked_scenes, parsed_query)
            
            # Step 9: Format results according to requirements
            results = self._format_results(ranked_scenes, rag_answer, use_rag)
            
            logger.info(f"Search completed. Found {len(ranked_scenes)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search pipeline failed: {str(e)}")
            raise
    
    async def _parse_and_expand_query(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Step 1: Parse and expand Vietnamese query using LLM
        """
        logger.info("Parsing and expanding query...")
        
        # Use LLM service for query parsing
        parsed_query = await llm_service.parse_and_expand_query(query, filters)
        
        logger.info(f"Parsed query: {parsed_query}")
        return parsed_query
    
    async def _process_audio_query(self, audio_bytes: bytes) -> str:
        """Process audio query using ASR"""
        logger.info("Processing audio query with ASR...")
        
        # Convert audio to text using ASR
        transcript = await self.asr_model.transcribe(audio_bytes)
        
        # Normalize the transcript
        normalized_transcript = text_normalizer.normalize_text(transcript)
        
        logger.info(f"Audio query transcribed: {transcript}")
        return normalized_transcript
    
    async def _pre_filter_metadata(self, parsed_query: Dict[str, Any]) -> List[str]:
        """
        Step 2: Pre-filter metadata in Elasticsearch
        """
        logger.info("Pre-filtering metadata...")
        
        # Extract filters from parsed query
        filters = parsed_query.get("extracted_filters", {})
        keywords = parsed_query.get("keywords", [])
        
        # Build Elasticsearch query
        es_query = self._build_es_query(filters, keywords)
        
        # Execute search
        filtered_scene_ids = await self.metadata_db.search_scenes(es_query)
        
        logger.info(f"Pre-filtered {len(filtered_scene_ids)} scenes")
        return filtered_scene_ids
    
    def _build_es_query(self, filters: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Build Elasticsearch query from filters and keywords"""
        query_parts = []
        
        # Add keyword search
        if keywords:
            keyword_query = {
                "multi_match": {
                    "query": " ".join(keywords),
                    "fields": ["transcript", "ocr_text", "scene_description"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            query_parts.append(keyword_query)
        
        # Add filter queries
        if filters.get("objects"):
            object_query = {
                "terms": {
                    "detected_objects": filters["objects"]
                }
            }
            query_parts.append(object_query)
        
        if filters.get("location"):
            location_query = {
                "match": {
                    "location": filters["location"]
                }
            }
            query_parts.append(location_query)
        
        if filters.get("time_of_day"):
            time_query = {
                "match": {
                    "time_of_day": filters["time_of_day"]
                }
            }
            query_parts.append(time_query)
        
        # Combine queries
        if len(query_parts) == 1:
            return {"query": query_parts[0]}
        else:
            return {
                "query": {
                    "bool": {
                        "must": query_parts
                    }
                }
            }
    
    async def _encode_query(
        self, 
        parsed_query: Dict[str, Any],
        query_type: str,
        query_image: Optional[bytes] = None,
        query_audio: Optional[bytes] = None
    ) -> Dict[str, np.ndarray]:
        """
        Step 3: Encode query to embeddings for specific modality
        """
        logger.info("Encoding query to embeddings...")
        
        embeddings = {}
        
        if query_type == "text":
            # Text embedding using PhoBERT (semantic)
            if parsed_query.get("normalized_query"):
                text_embedding = await self.embedding_model.encode_text(
                    parsed_query["normalized_query"]
                )
                embeddings["text"] = text_embedding
                # CLIP text embedding for cross-modal search into image index
                clip_text_embedding = await self.embedding_model.encode_clip_text(
                    parsed_query["normalized_query"]
                )
                embeddings["image"] = clip_text_embedding
        
        elif query_type == "image":
            # Image embedding using CLIP
            if query_image:
                image_embedding = await self.embedding_model.encode_image(query_image)
                embeddings["image"] = image_embedding
        
        elif query_type == "audio":
            # Audio embedding using Wav2Vec2
            if query_audio:
                audio_embedding = await self.embedding_model.encode_audio(query_audio)
                embeddings["audio"] = audio_embedding
        
        logger.info(f"Generated embeddings for modality: {list(embeddings.keys())}")
        return embeddings
    
    async def _fusion_search(
        self, 
        query_embeddings: Dict[str, np.ndarray],
        query_type: str,
        filtered_scene_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Fusion search (primary) with fallback to separate modalities
        """
        logger.info("Performing fusion search...")
        
        # Try fusion search first
        fusion_results = await self.vector_db.search_fusion(
            query_embeddings, query_type, filtered_scene_ids, top_k
        )
        
        if fusion_results:
            logger.info(f"Fusion search returned {len(fusion_results)} results")
            return fusion_results
        else:
            # Fallback to separate modality search
            logger.info("Fusion search returned no results, falling back to separate modalities")
            return await self._ann_search_separate_modalities(
                query_embeddings, filtered_scene_ids, top_k
            )
    
    async def _ann_search_separate_modalities(
        self, 
        query_embeddings: Dict[str, np.ndarray],
        filtered_scene_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback: ANN search using separate modality indices
        """
        logger.info("Performing ANN search with separate modality indices...")
        
        candidate_scenes = []
        
        # Search in each modality-specific index
        for modality, embedding in query_embeddings.items():
            modality_results = await self.vector_db.search_modality(
                modality, embedding, filtered_scene_ids, top_k
            )
            candidate_scenes.extend(modality_results)
        
        # Remove duplicates and sort by score
        unique_scenes = self._deduplicate_scenes(candidate_scenes)
        
        logger.info(f"Separate modality search found {len(unique_scenes)} candidate scenes")
        return unique_scenes
    
    def _deduplicate_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate scenes and sort by score"""
        seen_scenes = set()
        unique_scenes = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            if scene_id not in seen_scenes:
                seen_scenes.add(scene_id)
                unique_scenes.append(scene)
        
        # Sort by score (higher is better)
        unique_scenes.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique_scenes
    
    async def _enrich_scenes_with_metadata(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Enrich scenes with metadata from Elasticsearch for re-ranking
        """
        logger.info("Enriching scenes with metadata...")
        
        enriched_scenes = []
        for scene in scenes:
            scene_id = scene.get("scene_id")
            if scene_id:
                # Get metadata from Elasticsearch
                metadata = await self.metadata_db.get_scene_metadata(scene_id)
                if metadata:
                    # Merge metadata with scene data
                    scene.update(metadata)
                enriched_scenes.append(scene)
        
        logger.info(f"Enriched {len(enriched_scenes)} scenes with metadata")
        return enriched_scenes
    
    async def _re_rank_scenes(
        self, 
        candidate_scenes: List[Dict[str, Any]],
        parsed_query: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Step 6: Re-rank scenes using Vietnamese cross-encoder + fuzzy text match + quality score
        """
        logger.info("Re-ranking candidate scenes...")
        
        # Use Vietnamese cross-encoder for re-ranking
        original_query = parsed_query.get("original_query", "")
        re_ranked_scenes = self.cross_encoder.re_rank_scenes(original_query, candidate_scenes)
        
        # Calculate additional scores
        for scene in re_ranked_scenes:
            # Calculate fuzzy text match score
            fuzzy_score = self._calculate_fuzzy_text_score(parsed_query, scene)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(scene)
            
            # Get cross-encoder score (already calculated)
            cross_encoder_score = scene.get("cross_encoder_score", 0.5)
            
            # Combine scores with weights
            final_score = (
                cross_encoder_score * 0.6 +  # Higher weight for cross-encoder
                fuzzy_score * 0.25 +
                quality_score * 0.15
            )
            
            scene["final_score"] = final_score
            scene["fuzzy_score"] = fuzzy_score
            scene["quality_score"] = quality_score
        
        # Sort by final score
        re_ranked_scenes.sort(key=lambda x: x["final_score"], reverse=True)
        
        logger.info(f"Re-ranked {len(re_ranked_scenes)} scenes")
        return re_ranked_scenes[:top_k]
    
    def _calculate_fuzzy_text_score(
        self, 
        parsed_query: Dict[str, Any], 
        scene: Dict[str, Any]
    ) -> float:
        """Calculate fuzzy text matching score"""
        query_keywords = parsed_query.get("keywords", [])
        scene_text = scene.get("transcript", "") + " " + scene.get("ocr_text", "")
        
        if not query_keywords or not scene_text:
            return 0.0
        
        # Calculate fuzzy match score for each keyword
        total_score = 0.0
        for keyword in query_keywords:
            if text_normalizer.fuzzy_match(keyword, scene_text, threshold=0.7):
                total_score += 1.0
        
        return total_score / len(query_keywords) if query_keywords else 0.0
    
    def _calculate_quality_score(self, scene: Dict[str, Any]) -> float:
        """Calculate quality score based on scene characteristics"""
        quality_score = 0.5  # Base score
        
        # Higher score for scenes with more information
        if scene.get("transcript"):
            quality_score += 0.2
        if scene.get("ocr_text"):
            quality_score += 0.1
        if scene.get("detected_objects"):
            quality_score += 0.1
        if scene.get("scene_description"):
            quality_score += 0.1
        
        # Higher score for longer scenes (more content)
        duration = scene.get("duration", 0)
        if duration > 5.0:
            quality_score += 0.1
        elif duration > 2.0:
            quality_score += 0.05
        
        return min(quality_score, 1.0)
    
    async def _generate_rag_answer(
        self, 
        ranked_scenes: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> Optional[str]:
        """
        Step 7: Generate RAG answer using OpenAI
        """
        if not ranked_scenes:
            return None
        
        logger.info("Generating RAG answer...")
        
        # Use LLM service for RAG
        original_query = parsed_query.get("original_query", "")
        rag_answer = await llm_service.generate_rag_answer(ranked_scenes, original_query)
        
        return rag_answer
    
    def _format_results(
        self, 
        ranked_scenes: List[Dict[str, Any]], 
        rag_answer: Optional[str],
        use_rag: bool
    ) -> Dict[str, Any]:
        """
        Step 8: Format final results according to requirements
        """
        logger.info("Formatting search results...")
        
        # Format scenes according to requirements
        formatted_results = []
        for scene in ranked_scenes:
            # Extract video name and frame index
            video_id = scene.get("video_id", "")
            frame_idx = scene.get("frame_idx") or scene.get("idx_frame", 0)
            
            # Create video filename
            video_filename = f"{video_id}.mp4" if video_id else "unknown.mp4"
            
            result = {
                "video_name": video_filename,
                "frame_idx": int(frame_idx),
                "score": scene.get("final_score", 0.0),
                "start_time": scene.get("start_time", 0.0),
                "metadata": {
                    "transcript": scene.get("transcript", ""),
                    "ocr_text": scene.get("ocr_text", ""),
                    "detected_objects": scene.get("detected_objects", []),
                    "cross_encoder_score": scene.get("cross_encoder_score", 0.0),
                    "fuzzy_score": scene.get("fuzzy_score", 0.0),
                    "quality_score": scene.get("quality_score", 0.0)
                }
            }
            formatted_results.append(result)
        
        # Build response according to requirements
        results = {
            "query": "",  # Will be set by API layer
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_time": 0.0,  # Will be set by API layer
        }
        
        # Add RAG answer if requested and available
        if use_rag and rag_answer:
            results["answer"] = rag_answer
        
        return results


# Global search service instance
search_service = SearchService()
