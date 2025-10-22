"""
Vector Database for AIC2025 using FAISS

This module provides functionality to build, save, and query FAISS indexes
for frame-level and object-level embeddings.
"""

import os
import json
import numpy as np
import faiss
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_db")

class VideoEmbeddingIndex:
    """
    A FAISS-based vector database for video embeddings.
    Handles both frame-level and object-level embeddings.
    """
    
    def __init__(
        self, 
        data_root: str = "../data",
        frame_dim: int = 512,
        obj_dim: int = 512,
        use_gpu: bool = False
    ):
        """
        Initialize the vector database.
        
        Args:
            data_root: Root directory containing the embeddings data
            frame_dim: Dimension of frame embeddings
            obj_dim: Dimension of object embeddings
            use_gpu: Whether to use GPU for FAISS
        """
        self.data_root = Path(data_root).resolve()
        self.frame_dim = frame_dim
        self.obj_dim = obj_dim
        self.use_gpu = use_gpu
        
        # Embeddings paths
        self.embed_root = self.data_root / "embeddings_v2"
        self.frame_embed_dir = self.embed_root / "embeds" / "frame_clip"
        self.obj_embed_dir = self.embed_root / "embeds" / "obj_clip"
        
        # Metadata paths
        self.frame_meta_dir = self.embed_root / "meta" / "frame_meta"
        self.obj_meta_dir = self.embed_root / "meta" / "obj_meta"
        self.frame_index_path = self.embed_root / "frame_index.json"
        self.obj_index_path = self.embed_root / "obj_index.json"
        
        # FAISS indexes
        self.frame_index = None
        self.obj_index = None
        
        # Metadata lookup tables
        self.frame_metadata = {}  # {uid: metadata}
        self.obj_metadata = {}    # {uid: metadata}
        
        # ID mapping
        self.frame_id_to_uid = {}  # {faiss_id: uid}
        self.obj_id_to_uid = {}    # {faiss_id: uid}
        
        # Initialize GPU resources if needed
        self.res = None
        if use_gpu:
            try:
                # Check if GPU functions are available
                if hasattr(faiss, 'get_num_gpus') and hasattr(faiss, 'StandardGpuResources'):
                    gpu_count = faiss.get_num_gpus()
                    if gpu_count > 0:
                        self.res = faiss.StandardGpuResources()
                        logger.info(f"GPU acceleration enabled. Found {gpu_count} GPU(s)")
                    else:
                        logger.warning("No GPUs found, falling back to CPU")
                        self.use_gpu = False
                else:
                    logger.warning("FAISS GPU support not available (using faiss-cpu package)")
                    logger.warning("Install faiss-gpu for GPU acceleration")
                    self.use_gpu = False
            except Exception as e:
                logger.warning(f"Failed to initialize GPU resources: {e}")
                logger.warning("Falling back to CPU")
                self.use_gpu = False
        else:
            logger.info("Using CPU for FAISS")
    
    def build_indexes(self, videos: Optional[List[str]] = None) -> None:
        """
        Build FAISS indexes for both frame and object embeddings.
        
        Args:
            videos: Optional list of video IDs to include. If None, all videos are processed.
        """
        logger.info("Building FAISS indexes...")
        start_time = datetime.now()
        
        # Load global metadata indexes
        self._load_metadata(videos)
        
        # Build frame index
        self._build_frame_index()
        
        # Build object index
        self._build_object_index()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Indexes built in {elapsed:.2f} seconds")
    
    def _load_metadata(self, videos: Optional[List[str]] = None) -> None:
        """Load metadata for frames and objects."""
        logger.info("Loading metadata...")
        
        # Frame metadata
        with open(self.frame_index_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)
        
        # Object metadata
        # with open(self.obj_index_path, 'r', encoding='utf-8') as f:
        #     obj_data = json.load(f)
        
        # Filter by video IDs
        if videos:
            frame_data = [item for item in frame_data if item.get('video_id') in videos]
            # obj_data = [item for item in obj_data if item.get('video_id') in videos]
        
        # Check required keys
        required_keys = ['uid', 'video_id', 'frame_idx']
        valid_frame_data = []
        for item in frame_data:
            if all(key in item for key in required_keys):
                valid_frame_data.append(item)
            else:
                missing_keys = [key for key in required_keys if key not in item]
                logger.warning(f"Frame item missing required keys {missing_keys}: {item}")
        
        # Check required keys
        # required_keys = ['uid', 'video_id', 'frame_idx', 'obj_id']
        # valid_obj_data = []
        # for item in obj_data:
        #     if all(key in item for key in required_keys):
        #         valid_obj_data.append(item)
        #     else:
        #         missing_keys = [key for key in required_keys if key not in item]
        #         logger.warning(f"Object item missing required keys {missing_keys}: {item}")
        
        # Build lookup tables
        self.frame_metadata = {item['uid']: item for item in valid_frame_data}
        # self.obj_metadata = {item['uid']: item for item in valid_obj_data}
        
        # Log a sample metadata entry for debugging
        if self.frame_metadata:
            sample_uid = next(iter(self.frame_metadata))
            logger.info(f"Sample frame metadata: {self.frame_metadata[sample_uid]}")
        
        # if self.obj_metadata:
        #     sample_uid = next(iter(self.obj_metadata))
        #     logger.info(f"Sample object metadata: {self.obj_metadata[sample_uid]}")
        
        logger.info(f"Loaded metadata for {len(self.frame_metadata)} frames and {len(self.obj_metadata)} objects")
    
    def _build_frame_index(self) -> None:
        """Build FAISS index for frame embeddings."""
        logger.info("Building frame embedding index...")
        
        # Initialize index (L2 distance)
        frame_index = faiss.IndexFlatL2(self.frame_dim)
        
        # Move to GPU if available and GPU functions are available
        if self.use_gpu and self.res and hasattr(faiss, 'index_cpu_to_gpu'):
            try:
                frame_index = faiss.index_cpu_to_gpu(self.res, 0, frame_index)
                logger.info("Using GPU for frame index")
            except Exception as e:
                logger.warning(f"Failed to move frame index to GPU: {e}")
                logger.warning("Continuing with CPU index")
        
        # Collect embeddings
        frame_embeddings = []
        frame_uids = []
        
        sample_dims = set()  # Track dimensions of the embeddings we encounter
        
        for uid, metadata in self.frame_metadata.items():
            video_id = metadata['video_id']
            frame_idx = metadata['frame_idx']
            
            # Get embedding path
            embed_path = self.frame_embed_dir / video_id / f"{frame_idx}.npy"
            
            if embed_path.exists():
                embedding = np.load(embed_path)
                sample_dims.add(embedding.shape[0])
                
                # Ensure correct dimension
                if embedding.shape[0] != self.frame_dim:
                    logger.warning(f"Frame embedding dimension mismatch: {embedding.shape[0]} != {self.frame_dim}")
                    
                    # Adapt dimension
                    if embedding.shape[0] < self.frame_dim:
                        # Pad with zeros
                        pad_size = self.frame_dim - embedding.shape[0]
                        embedding = np.pad(embedding, (0, pad_size), 'constant')
                        logger.info(f"Padded embedding from {embedding.shape[0]-pad_size} to {embedding.shape[0]}")
                    else:
                        # Truncate
                        embedding = embedding[:self.frame_dim]
                        logger.info(f"Truncated embedding to {self.frame_dim}")
                
                frame_embeddings.append(embedding)
                frame_uids.append(uid)
                
        logger.info(f"Sample embedding dimensions encountered: {sample_dims}")
        
        # Create ID mapping
        self.frame_id_to_uid = {i: uid for i, uid in enumerate(frame_uids)}
        
        # Add to index
        if frame_embeddings:
            frame_embeddings = np.vstack(frame_embeddings).astype(np.float32)
            # Debug: log pre-normalization norm stats
            try:
                pre_norm_mean = float(np.mean(np.linalg.norm(frame_embeddings, axis=1)))
                logger.info(f"Frame embeddings pre-normalization mean norm: {pre_norm_mean:.4f}")
            except Exception:
                pass
            # L2-normalize embeddings row-wise for cosine via L2
            norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True) + 1e-12
            frame_embeddings = frame_embeddings / norms
            try:
                post_norm_mean = float(np.mean(np.linalg.norm(frame_embeddings, axis=1)))
                logger.info(f"Frame embeddings post-normalization mean norm: {post_norm_mean:.4f}")
            except Exception:
                pass
            frame_index.add(frame_embeddings)
            
            logger.info(f"Added {len(frame_embeddings)} frame embeddings to index")
        else:
            logger.warning("No frame embeddings found")
        
        self.frame_index = frame_index
    
    def _build_object_index(self) -> None:
        """Build FAISS index for object embeddings."""
        logger.info("Building object embedding index...")
        
        # Initialize index (L2 distance)
        obj_index = faiss.IndexFlatL2(self.obj_dim)
        
        # Move to GPU if available and GPU functions are available
        if self.use_gpu and self.res and hasattr(faiss, 'index_cpu_to_gpu'):
            try:
                obj_index = faiss.index_cpu_to_gpu(self.res, 0, obj_index)
                logger.info("Using GPU for object index")
            except Exception as e:
                logger.warning(f"Failed to move object index to GPU: {e}")
                logger.warning("Continuing with CPU index")
        
        # Collect embeddings
        obj_embeddings = []
        obj_uids = []
        
        sample_dims = set()  # Track dimensions of the embeddings we encounter
        
        for uid, metadata in self.obj_metadata.items():
            if not metadata.get('crop_available', False):
                continue  # Skip objects without crop images
                
            video_id = metadata['video_id']
            frame_idx = metadata['frame_idx']
            obj_id = metadata['obj_id']
            
            # Get embedding path
            embed_path = self.obj_embed_dir / video_id / f"{frame_idx}__{obj_id}.npy"
            
            if embed_path.exists():
                embedding = np.load(embed_path)
                sample_dims.add(embedding.shape[0])
                
                # Ensure correct dimension
                if embedding.shape[0] != self.obj_dim:
                    logger.warning(f"Object embedding dimension mismatch: {embedding.shape[0]} != {self.obj_dim}")
                    
                    # Adapt dimension
                    if embedding.shape[0] < self.obj_dim:
                        # Pad with zeros
                        pad_size = self.obj_dim - embedding.shape[0]
                        embedding = np.pad(embedding, (0, pad_size), 'constant')
                        logger.info(f"Padded embedding from {embedding.shape[0]-pad_size} to {embedding.shape[0]}")
                    else:
                        # Truncate
                        embedding = embedding[:self.obj_dim]
                        logger.info(f"Truncated embedding to {self.obj_dim}")
                
                obj_embeddings.append(embedding)
                obj_uids.append(uid)
        
        logger.info(f"Sample object embedding dimensions encountered: {sample_dims}")
        
        # Create ID mapping
        self.obj_id_to_uid = {i: uid for i, uid in enumerate(obj_uids)}
        
        # Add to index
        if obj_embeddings:
            obj_embeddings = np.vstack(obj_embeddings).astype(np.float32)
            # Debug: log pre-normalization norm stats
            try:
                pre_norm_mean = float(np.mean(np.linalg.norm(obj_embeddings, axis=1)))
                logger.info(f"Object embeddings pre-normalization mean norm: {pre_norm_mean:.4f}")
            except Exception:
                pass
            # L2-normalize embeddings row-wise for cosine via L2
            norms = np.linalg.norm(obj_embeddings, axis=1, keepdims=True) + 1e-12
            obj_embeddings = obj_embeddings / norms
            try:
                post_norm_mean = float(np.mean(np.linalg.norm(obj_embeddings, axis=1)))
                logger.info(f"Object embeddings post-normalization mean norm: {post_norm_mean:.4f}")
            except Exception:
                pass
            obj_index.add(obj_embeddings)
            
            logger.info(f"Added {len(obj_embeddings)} object embeddings to index")
        else:
            logger.warning("No object embeddings found")
        
        self.obj_index = obj_index
    
    def save_indexes(self, output_dir: Union[str, Path]) -> None:
        """
        Save FAISS indexes to disk.
        
        Args:
            output_dir: Directory to save indexes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving indexes to {output_dir}")
        
        # Save frame index
        if self.frame_index:
            # Convert to CPU index if on GPU
            index_to_save = self.frame_index
            # Only convert if GPU index and GPU-to-CPU conversion is available
            if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                index_to_save = faiss.index_gpu_to_cpu(self.frame_index)
            
            faiss.write_index(index_to_save, str(output_dir / "frame_index.faiss"))
            
            # Save ID mapping
            with open(output_dir / "frame_id_mapping.json", 'w', encoding='utf-8') as f:
                json.dump(self.frame_id_to_uid, f, ensure_ascii=False)
        
        # Save object index
        if self.obj_index:
            # Convert to CPU index if on GPU
            index_to_save = self.obj_index
            # Only convert if GPU index and GPU-to-CPU conversion is available
            if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                index_to_save = faiss.index_gpu_to_cpu(self.obj_index)
            
            faiss.write_index(index_to_save, str(output_dir / "obj_index.faiss"))
            
            # Save ID mapping
            with open(output_dir / "obj_id_mapping.json", 'w', encoding='utf-8') as f:
                json.dump(self.obj_id_to_uid, f, ensure_ascii=False)
        
        logger.info("Indexes saved successfully")
    
    def load_indexes(self, index_dir: Union[str, Path]) -> None:
        """
        Load FAISS indexes from disk.
        
        Args:
            index_dir: Directory containing saved indexes
        """
        index_dir = Path(index_dir)
        logger.info(f"Loading indexes from {index_dir}")
        
        # Load frame index
        frame_index_path = index_dir / "frame_clip_index.faiss"
        if frame_index_path.exists():
            self.frame_index = faiss.read_index(str(frame_index_path))
            
            # Move to GPU if needed and GPU functions are available
            if self.use_gpu and self.res and hasattr(faiss, 'index_cpu_to_gpu'):
                try:
                    self.frame_index = faiss.index_cpu_to_gpu(self.res, 0, self.frame_index)
                except Exception as e:
                    logger.warning(f"Failed to move frame index to GPU: {e}")
                    logger.warning("Continuing with CPU index")
            
            # Load ID mapping
            with open(index_dir / "frame_clip_id_mapping.json", 'r', encoding='utf-8') as f:
                # JSON keys are strings, convert back to integers
                self.frame_id_to_uid = {int(k): v for k, v in json.load(f).items()}
            
            logger.info(f"Loaded frame index with {self.frame_index.ntotal} vectors")
        else:
            logger.warning("Frame index not found")
        
        # Load object index
        obj_index_path = index_dir / "obj_clip_index.faiss"
        if obj_index_path.exists():
            self.obj_index = faiss.read_index(str(obj_index_path))
            
            # Move to GPU if needed and GPU functions are available
            if self.use_gpu and self.res and hasattr(faiss, 'index_cpu_to_gpu'):
                try:
                    self.obj_index = faiss.index_cpu_to_gpu(self.res, 0, self.obj_index)
                except Exception as e:
                    logger.warning(f"Failed to move object index to GPU: {e}")
                    logger.warning("Continuing with CPU index")
            
            # Load ID mapping
            with open(index_dir / "obj_clip_id_mapping.json", 'r', encoding='utf-8') as f:
                # JSON keys are strings, convert back to integers
                self.obj_id_to_uid = {int(k): v for k, v in json.load(f).items()}
            
            logger.info(f"Loaded object index with {self.obj_index.ntotal} vectors")
        else:
            logger.warning("Object index not found")

        # Ensure metadata lookup tables are populated after loading indexes
        try:
            self._load_metadata()
            logger.info("Metadata loaded successfully after index load")
        except Exception as e:
            logger.warning(f"Failed to load metadata after index load: {e}")
    
    def search_by_text(
        self, 
        query: str, 
        index_type: str = "frame", 
        top_k: int = 10,
        embedding_model = None
    ) -> List[Dict[str, Any]]:
        """
        Search frames or objects by text query.
        
        Args:
            query: Text query
            index_type: Type of index to search ("frame" or "object")
            top_k: Number of results to return
            embedding_model: Models dictionary from init_models()
            
        Returns:
            List of results with metadata and scores
        """
        if embedding_model is None:
            raise ValueError("Embedding model is required for text search")
        
        # Import these functions here to avoid circular imports
        from embedding_model import embed_vietnamese_text, embed_clip_text
        
        # Embed the query
        if index_type == "frame":
            # For frame index, assume text is Vietnamese and needs PhoBERT embedding
            base_embedding = embed_vietnamese_text(query, embedding_model)
            
            # PhoBERT outputs 768-dim vector, but our frame index expects 1024-dim vector
            # Adapt the dimension (simple padding for now, could be improved)
            if self.frame_dim > len(base_embedding):
                # Pad with zeros
                pad_size = self.frame_dim - len(base_embedding)
                query_embedding = np.pad(base_embedding, (0, pad_size), 'constant')
            elif self.frame_dim < len(base_embedding):
                # Truncate
                query_embedding = base_embedding[:self.frame_dim]
            else:
                query_embedding = base_embedding
                
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            return self.search_by_vector(query_embedding, index_type, top_k)
        else:
            # For object index, use CLIP text embedding
            base_embedding = embed_clip_text(query, embedding_model)
            
            # CLIP outputs 512-dim vector, but adapt if our object index has different dimension
            if self.obj_dim > len(base_embedding):
                # Pad with zeros
                pad_size = self.obj_dim - len(base_embedding)
                query_embedding = np.pad(base_embedding, (0, pad_size), 'constant')
            elif self.obj_dim < len(base_embedding):
                # Truncate
                query_embedding = base_embedding[:self.obj_dim]
            else:
                query_embedding = base_embedding
                
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            return self.search_by_vector(query_embedding, index_type, top_k)

    def search_by_text_fusion(
        self,
        query: str,
        top_k: int = 10,
        embedding_model = None,
        asr_conf: float = 0.0,  # Default to 0 for OCR+Image only
        ocr_conf: float = 0.8,  # Default to 0.8 for OCR+Image only
        scene_score: float = 0.0,  # Default to 0 for OCR+Image only
        has_objects: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search frame index using a multimodal fused query vector that mirrors frame fusion.

        The query is embedded into:
        - ASR (PhoBERT) using the raw text (disabled by default for OCR+Image setup)
        - OCR: zero vector by default (no boxes for a pure text query)
        - Scene (PhoBERT) using the raw text with a scene_score weight (disabled by default)
        - Global CLIP text embedding

        Then the parts are fused with the same projection as frames (to 1024-d),
        using adaptive weights computed from provided confidence hints.
        Default setup: OCR + Image only for better accuracy.
        """
        if embedding_model is None:
            raise ValueError("Embedding model is required for fusion text search")

        # Import here to avoid circular deps
        from embedding_model import (
            embed_vietnamese_text,
            embed_clip_text,
            create_frame_fusion_vector,
            compute_adaptive_weights,
            l2_normalize,
        )

        # Build per-modality embeddings
        # ASR embedding: use text query if asr_conf > 0, otherwise zero vector
        if asr_conf > 0:
            asr_emb = embed_vietnamese_text(query, embedding_model)  # 768
        else:
            asr_emb = np.zeros(768, dtype=np.float32)
        
        # OCR embedding: use text query if ocr_conf > 0, otherwise zero vector
        if ocr_conf > 0:
            # For text queries, we can't use layout-aware OCR, so use text embedding
            ocr_emb = embed_vietnamese_text(query, embedding_model)  # 768
        else:
            ocr_emb = np.zeros(768, dtype=np.float32)
        
        # Scene embedding: use text query if scene_score > 0, otherwise zero vector
        if scene_score > 0:
            scene_text_emb = embed_vietnamese_text(query, embedding_model)  # 768
            scene_emb = scene_text_emb * float(max(0.0, min(1.0, scene_score)))
        else:
            scene_emb = np.zeros(768, dtype=np.float32)
        
        # CLIP text
        gclip_emb = embed_clip_text(query, embedding_model)  # 512

        # Adaptive weights using provided confidences/hints
        weights = compute_adaptive_weights(
            float(max(0.0, min(1.0, asr_conf))),
            float(max(0.0, min(1.0, ocr_conf))),
            float(max(0.0, min(1.0, scene_score))),
            bool(has_objects),
        )

        # Debug: log modality norms and weights
        try:
            asr_norm = float(np.linalg.norm(asr_emb)) if asr_emb is not None else -1.0
            ocr_norm = float(np.linalg.norm(ocr_emb)) if ocr_emb is not None else -1.0
            scene_norm = float(np.linalg.norm(scene_emb)) if scene_emb is not None else -1.0
            gclip_norm = float(np.linalg.norm(gclip_emb)) if gclip_emb is not None else -1.0
            logger.info(
                f"Fusion debug | params(asr_conf={asr_conf}, ocr_conf={ocr_conf}, scene_score={scene_score}, has_objects={has_objects}) "
                f"| norms(asr={asr_norm:.4f}, ocr={ocr_norm:.4f}, scene={scene_norm:.4f}, gclip={gclip_norm:.4f}) "
                f"| weights={weights}"
            )
        except Exception as e:
            logger.warning(f"Failed to log fusion debug info: {e}")

        # Fuse to 1024-d to match frame index
        # fused_query = create_frame_fusion_vector(
        #     asr_emb=asr_emb,
        #     ocr_emb=ocr_emb,
        #     scene_emb=scene_emb,
        #     gclip_emb=gclip_emb,
        #     weights=weights,
        #     target_dim=self.frame_dim,
        # )
        fused_query = gclip_emb

        # Debug: fused query norm
        try:
            fused_norm = float(np.linalg.norm(fused_query)) if fused_query is not None else -1.0
            logger.info(f"Fusion debug | fused_query_norm={fused_norm:.4f}, target_dim={self.frame_dim}")
        except Exception as e:
            logger.warning(f"Failed to log fused query norm: {e}")

        fused_query = fused_query.reshape(1, -1).astype(np.float32)
        return self.search_by_vector(fused_query, index_type="frame", top_k=top_k)

    def search_by_text_fusion_v2(
        self,
        query: str,
        top_k: int = 10,
        embedding_model = None,
        clip_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Search CLIP-only (512d) frame index using CLIP text embedding.
        Simplified version for v2 embeddings.
        """
        if embedding_model is None:
            raise ValueError("Embedding model is required for CLIP text search")

        # Import here to avoid circular deps
        from embedding_model_2 import embed_clip_text, l2_normalize

        # CLIP text embedding only (512d)
        clip_emb = embed_clip_text(query, embedding_model)  # 512d
        
        # Apply weight if needed
        if clip_weight != 1.0:
            clip_emb = clip_emb * float(max(0.0, min(1.0, clip_weight)))
        
        # Debug: log CLIP norm before normalization
        try:
            clip_norm = float(np.linalg.norm(clip_emb))
            logger.info(f"CLIP-only debug | clip_weight={clip_weight} | clip_norm_before={clip_norm:.4f}")
        except Exception as e:
            logger.warning(f"Failed to log CLIP debug info: {e}")

        clip_emb = clip_emb.reshape(1, -1).astype(np.float32)
        return self.search_by_vector(clip_emb, index_type="frame", top_k=top_k)
    
    def search_by_vector(
        self, 
        query_vector: np.ndarray, 
        index_type: str = "frame", 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search frames or objects by vector query.
        
        Args:
            query_vector: Embedding vector
            index_type: Type of index to search ("frame" or "object")
            top_k: Number of results to return
            
        Returns:
            List of results with metadata and scores
        """
        if index_type == "frame":
            if self.frame_index is None:
                raise ValueError("Frame index not built or loaded")
            
            index = self.frame_index
            id_mapping = self.frame_id_to_uid
            metadata_lookup = self.frame_metadata
            logger.info(f"Frame index dimension: {self.frame_dim}, Query vector dimension: {query_vector.shape}")
        else:  # object
            if self.obj_index is None:
                raise ValueError("Object index not built or loaded")
            
            index = self.obj_index
            id_mapping = self.obj_id_to_uid
            metadata_lookup = self.obj_metadata
            logger.info(f"Object index dimension: {self.obj_dim}, Query vector dimension: {query_vector.shape}")
        # Ensure float32 and L2-normalize the query vector(s)
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # L2 normalization per query in the batch
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-12
        query_vector = query_vector / norms

        # Perform search
        distances, indices = index.search(query_vector, top_k)
        
        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for not enough results
                continue
                
            uid = id_mapping.get(idx)
            if uid is None:
                logger.warning(f"No UID found for index {idx}")
                continue
            
            metadata = metadata_lookup.get(uid, {})
            if not metadata:
                logger.warning(f"No metadata found for UID {uid}")
                
            # Log the metadata structure for debugging
            logger.debug(f"Metadata for UID {uid}: {metadata}")
            
            # Simple and effective similarity score
            # Works well for both normalized and unnormalized vectors
            score = float(1.0 / (1.0 + dist))
            
            results.append({
                "score": score,                     # Similarity score
                "distance": float(dist),            # L2 distance
                "rank": i + 1,
                "uid": uid,
                "metadata": metadata
            })
        
        return results
    
    def search_by_image(
        self, 
        image_path: str, 
        index_type: str = "frame", 
        top_k: int = 10,
        embedding_model = None
    ) -> List[Dict[str, Any]]:
        """
        Search frames or objects by image.
        
        Args:
            image_path: Path to image file
            index_type: Type of index to search ("frame" or "object")
            top_k: Number of results to return
            embedding_model: Models dictionary from init_models()
            
        Returns:
            List of results with metadata and scores
        """
        if embedding_model is None:
            raise ValueError("Embedding model is required for image search")
        
        # Import function here to avoid circular imports
        from embedding_model import embed_clip_image
        
        # Embed the image with CLIP
        base_embedding = embed_clip_image(image_path, embedding_model)
        
        # Adapt dimensions based on the target index
        target_dim = self.frame_dim if index_type == "frame" else self.obj_dim
        
        if target_dim > len(base_embedding):
            # Pad with zeros
            pad_size = target_dim - len(base_embedding)
            query_embedding = np.pad(base_embedding, (0, pad_size), 'constant')
        elif target_dim < len(base_embedding):
            # Truncate
            query_embedding = base_embedding[:target_dim]
        else:
            query_embedding = base_embedding
            
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        return self.search_by_vector(query_embedding, index_type, top_k)


def build_and_save_indexes(
    data_root: str = "../data",
    output_dir: str = "../data/faiss_indexes",
    videos: Optional[List[str]] = None,
    use_gpu: bool = False
) -> None:
    """
    Build and save FAISS indexes for all videos or specified videos.
    
    Args:
        data_root: Root directory containing the embeddings data
        output_dir: Directory to save indexes
        videos: Optional list of video IDs to include. If None, all videos are processed.
        use_gpu: Whether to use GPU for FAISS
    """
    # Create index builder
    index_builder = VideoEmbeddingIndex(
        data_root=data_root,
        frame_dim=1024,  # From embedding_model.py
        obj_dim=512,     # From embedding_model.py
        use_gpu=use_gpu
    )
    
    # Build indexes
    index_builder.build_indexes(videos)
    
    # Save indexes
    index_builder.save_indexes(output_dir)
    
    logger.info(f"FAISS indexes built and saved to {output_dir}")


# --------- V2: CLIP-only (512d) index builders and search helpers ---------
def _collect_v2_frame_entries(embeddings_root: Path, videos: Optional[List[str]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    meta_dir = embeddings_root / "meta" / "frame_meta"
    if not meta_dir.exists():
        return entries
    for video_dir in sorted(meta_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        if videos and video_id not in videos:
            continue
        for meta_file in sorted(video_dir.glob("*.json")):
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                uid = meta.get("uid")
                frame_idx = meta.get("frame_idx")
                if uid is None or frame_idx is None:
                    continue
                embed_path = embeddings_root / "embeds" / "frame_clip" / video_id / f"{frame_idx}.npy"
                if not embed_path.exists():
                    continue
                entries.append({
                    "uid": uid,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "embed_path": embed_path.as_posix(),
                    "meta_path": meta_file.as_posix(),
                })
            except Exception:
                continue
    return entries


def build_and_save_indexes_v2(
    embeddings_root: str = "data/embeddings_v2",
    output_dir: str = "data/faiss_index_v2",
    videos: Optional[List[str]] = None,
) -> None:
    """Build and save CLIP-only (512d) frame index for v2 embeddings."""
    embeddings_root_p = Path(embeddings_root)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building CLIP-only index from {embeddings_root_p}")
    entries = _collect_v2_frame_entries(embeddings_root_p, videos)
    # Ensure deterministic ordering by (video_id, frame_idx)
    try:
        entries = sorted(entries, key=lambda e: (e.get("video_id", ""), int(e.get("frame_idx", 0))))
    except Exception:
        pass
    if not entries:
        raise RuntimeError("No v2 frame entries found. Ensure embeddings_v2 exist.")

    vectors: List[np.ndarray] = []
    uids: List[str] = []
    for e in entries:
        vec = np.load(e["embed_path"]).astype(np.float32)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        if vec.shape[0] != 512:
            # Safety: pad/truncate
            if vec.shape[0] > 512:
                vec = vec[:512]
            else:
                vec = np.pad(vec, (0, 512 - vec.shape[0]), "constant")
        vectors.append(vec)
        uids.append(e["uid"])

    mat = np.vstack(vectors)
    # L2 normalize row-wise
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    try:
        mean_norm = float(np.mean(np.linalg.norm(mat, axis=1)))
        logger.info(f"V2 frame embeddings post-normalization mean norm: {mean_norm:.4f}")
    except Exception:
        pass

    # Build IndexFlatL2(512)
    index = faiss.IndexFlatL2(512)
    index.add(mat.astype(np.float32))

    # Save
    faiss.write_index(index, str(output_dir_p / "frame_clip_index.faiss"))
    id_map = {i: uid for i, uid in enumerate(uids)}
    with (output_dir_p / "frame_clip_id_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    logger.info(f"V2 index saved to {output_dir_p}, ntotal={index.ntotal}")

    # ---- Build object index (CLIP 512d) ----
    # try:
    #     obj_entries: List[Dict[str, Any]] = []
    #     obj_meta_root = embeddings_root_p / "meta" / "obj_meta"
    #     if obj_meta_root.exists():
    #         for vd in sorted(obj_meta_root.iterdir()):
    #             if not vd.is_dir():
    #                 continue
    #             video_id = vd.name
    #             if videos and video_id not in videos:
    #                 continue
    #             for mf in sorted(vd.glob("*.json")):
    #                 try:
    #                     with mf.open("r", encoding="utf-8") as f:
    #                         meta = json.load(f)
    #                     uid = meta.get("uid")
    #                     frame_idx = meta.get("frame_idx")
    #                     obj_id = meta.get("obj_id")
    #                     crop_available = bool(meta.get("crop_available", False))
    #                     if not (uid and frame_idx is not None and obj_id):
    #                         continue
    #                     # Embed path
    #                     embed_path = embeddings_root_p / "embeds" / "obj_crop" / video_id / f"{frame_idx}__{obj_id}.npy"
    #                     if not crop_available or not embed_path.exists():
    #                         # Skip objects without crop/image embedding
    #                         continue
    #                     obj_entries.append({
    #                         "uid": uid,
    #                         "video_id": video_id,
    #                         "frame_idx": int(frame_idx),
    #                         "obj_id": str(obj_id),
    #                         "embed_path": embed_path.as_posix(),
    #                     })
    #                 except Exception:
    #                     continue
    #     # Sort deterministically
    #     if obj_entries:
    #         try:
    #             obj_entries = sorted(obj_entries, key=lambda e: (e.get("video_id", ""), int(e.get("frame_idx", 0)), str(e.get("obj_id", ""))))
    #         except Exception:
    #             pass
    #         # Load embeddings
    #         obj_vecs: List[np.ndarray] = []
    #         obj_uids: List[str] = []
    #         for e in obj_entries:
    #             arr = np.load(e["embed_path"]).astype(np.float32)
    #             if arr.ndim != 1:
    #                 arr = arr.reshape(-1)
    #             if arr.shape[0] != 512:
    #                 if arr.shape[0] > 512:
    #                     arr = arr[:512]
    #                 else:
    #                     arr = np.pad(arr, (0, 512 - arr.shape[0]), "constant")
    #             obj_vecs.append(arr)
    #             obj_uids.append(e["uid"])
    #         obj_mat = np.vstack(obj_vecs)
    #         obj_mat = obj_mat / (np.linalg.norm(obj_mat, axis=1, keepdims=True) + 1e-12)
    #         obj_index = faiss.IndexFlatL2(512)
    #         obj_index.add(obj_mat.astype(np.float32))
    #         faiss.write_index(obj_index, str(output_dir_p / "obj_clip_index.faiss"))
    #         obj_id_map = {i: uid for i, uid in enumerate(obj_uids)}
    #         with (output_dir_p / "obj_clip_id_mapping.json").open("w", encoding="utf-8") as f:
    #             json.dump(obj_id_map, f, ensure_ascii=False)
    #         logger.info(f"V2 object index saved, ntotal={obj_index.ntotal}")
    #     else:
    #         logger.info("No v2 object entries found or crops missing; skipped object index build.")
    # except Exception as e:
    #     logger.warning(f"Failed to build v2 object index: {e}")


def search_text_clip_only(
    query: str,
    index_dir: str = "data/faiss_index_v2",
    embeddings_root: str = "data/embeddings_v2",
    top_k: int = 10,
    search_objects: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Search CLIP-only (512d) frame and object indexes using CLIP text embedding."""
    # Lazy import to avoid heavy deps globally
    from embedding_model_2 import init_models, embed_clip_text

    # Load frame index
    frame_index_path = Path(index_dir) / "frame_clip_index.faiss"
    frame_id_map_path = Path(index_dir) / "frame_clip_id_mapping.json"
    if not frame_index_path.exists() or not frame_id_map_path.exists():
        raise FileNotFoundError("V2 frame index or id mapping not found. Build with build_and_save_indexes_v2().")

    frame_index = faiss.read_index(str(frame_index_path))
    with frame_id_map_path.open("r", encoding="utf-8") as f:
        frame_id_map = {int(k): v for k, v in json.load(f).items()}

    # Load object index if available
    obj_index = None
    obj_id_map = {}
    if search_objects:
        obj_index_path = Path(index_dir) / "obj_clip_index.faiss"
        obj_id_map_path = Path(index_dir) / "obj_clip_id_mapping.json"
        if obj_index_path.exists() and obj_id_map_path.exists():
            obj_index = faiss.read_index(str(obj_index_path))
            with obj_id_map_path.open("r", encoding="utf-8") as f:
                obj_id_map = {int(k): v for k, v in json.load(f).items()}

    # Build metadata lookups
    frame_meta_lookup: Dict[str, Any] = {}
    obj_meta_lookup: Dict[str, Any] = {}
    frame_meta_root = Path(embeddings_root) / "meta" / "frame_meta"
    obj_meta_root = Path(embeddings_root) / "meta" / "obj_meta"
    
    if frame_meta_root.exists():
        for vd in frame_meta_root.iterdir():
            if not vd.is_dir():
                continue
            for mf in vd.glob("*.json"):
                try:
                    with mf.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    uid = meta.get("uid")
                    if uid:
                        frame_meta_lookup[uid] = meta
                except Exception:
                    continue
    
    if obj_meta_root.exists():
        for vd in obj_meta_root.iterdir():
            if not vd.is_dir():
                continue
            for mf in vd.glob("*.json"):
                try:
                    with mf.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    uid = meta.get("uid")
                    if uid:
                        obj_meta_lookup[uid] = meta
                except Exception:
                    continue

    models = init_models()
    q = embed_clip_text(query, models).astype(np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    # L2 normalize query
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    results = {"frames": [], "objects": []}
    
    # Search frames
    dists, inds = frame_index.search(q, top_k)
    for rank, (dist, idx) in enumerate(zip(dists[0], inds[0]), start=1):
        if idx < 0:
            continue
        uid = frame_id_map.get(int(idx))
        meta = frame_meta_lookup.get(uid, {})
        results["frames"].append({
            "score": float(1.0 / (1.0 + float(dist))),
            "distance": float(dist),
            "rank": rank,
            "uid": uid,
            "metadata": meta,
        })
    
    # Search objects if available
    if obj_index is not None:
        dists, inds = obj_index.search(q, top_k)
        for rank, (dist, idx) in enumerate(zip(dists[0], inds[0]), start=1):
            if idx < 0:
                continue
            uid = obj_id_map.get(int(idx))
            meta = obj_meta_lookup.get(uid, {})
            results["objects"].append({
                "score": float(1.0 / (1.0 + float(dist))),
                "distance": float(dist),
                "rank": rank,
                "uid": uid,
                "metadata": meta,
            })
    
    return results


def build_object_index_only_v2(
    embeddings_root: str = "data/embeddings_v2",
    output_dir: str = "data/faiss_index_v2",
    videos: Optional[List[str]] = None,
) -> None:
    """Build only object index for v2 embeddings (CLIP 512d).
    
    This function only builds the object index without touching the frame index.
    Useful when frame index is already built and you only need to build/rebuild object index.
    
    Args:
        embeddings_root: Root directory containing v2 embeddings
        output_dir: Output directory for the object index
        videos: Optional list of video IDs to process (if None, process all)
    """
    embeddings_root_p = Path(embeddings_root)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building CLIP-only object index from {embeddings_root_p}")
    
    # ---- Build object index (CLIP 512d) ----
    try:
        obj_entries: List[Dict[str, Any]] = []
        obj_meta_root = embeddings_root_p / "meta" / "obj_meta"
        if obj_meta_root.exists():
            for vd in sorted(obj_meta_root.iterdir()):
                if not vd.is_dir():
                    continue
                video_id = vd.name
                if videos and video_id not in videos:
                    continue
                for mf in sorted(vd.glob("*.json")):
                    try:
                        with mf.open("r", encoding="utf-8") as f:
                            meta = json.load(f)
                        uid = meta.get("uid")
                        frame_idx = meta.get("frame_idx")
                        obj_id = meta.get("obj_id")
                        crop_available = bool(meta.get("crop_available", False))
                        if not (uid and frame_idx is not None and obj_id):
                            continue
                        # Embed path
                        embed_path = embeddings_root_p / "embeds" / "obj_crop" / video_id / f"{frame_idx}__{obj_id}.npy"
                        if not crop_available or not embed_path.exists():
                            # Skip objects without crop/image embedding
                            continue
                        obj_entries.append({
                            "uid": uid,
                            "video_id": video_id,
                            "frame_idx": int(frame_idx),
                            "obj_id": str(obj_id),
                            "embed_path": embed_path.as_posix(),
                        })
                    except Exception:
                        continue
        
        # Sort deterministically
        if obj_entries:
            try:
                obj_entries = sorted(obj_entries, key=lambda e: (e.get("video_id", ""), int(e.get("frame_idx", 0)), str(e.get("obj_id", ""))))
            except Exception:
                pass
            
            logger.info(f"Found {len(obj_entries)} object entries to process")
            
            # Load embeddings
            obj_vecs: List[np.ndarray] = []
            obj_uids: List[str] = []
            for e in tqdm(obj_entries, desc="Loading object embeddings"):
                arr = np.load(e["embed_path"]).astype(np.float32)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                if arr.shape[0] != 512:
                    if arr.shape[0] > 512:
                        arr = arr[:512]
                    else:
                        arr = np.pad(arr, (0, 512 - arr.shape[0]), "constant")
                obj_vecs.append(arr)
                obj_uids.append(e["uid"])
            
            # Build index
            logger.info("Building object FAISS index...")
            obj_mat = np.vstack(obj_vecs)
            obj_mat = obj_mat / (np.linalg.norm(obj_mat, axis=1, keepdims=True) + 1e-12)
            obj_index = faiss.IndexFlatL2(512)
            obj_index.add(obj_mat.astype(np.float32))
            
            # Save index
            faiss.write_index(obj_index, str(output_dir_p / "obj_clip_index.faiss"))
            obj_id_map = {i: uid for i, uid in enumerate(obj_uids)}
            with (output_dir_p / "obj_clip_id_mapping.json").open("w", encoding="utf-8") as f:
                json.dump(obj_id_map, f, ensure_ascii=False)
            
            logger.info(f"✅ V2 object index saved to {output_dir_p}, ntotal={obj_index.ntotal}")
        else:
            logger.info("No v2 object entries found or crops missing; skipped object index build.")
    except Exception as e:
        logger.error(f"Failed to build v2 object index: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS indexes for video embeddings")
    parser.add_argument("--data-root", type=str, default="../data", help="Root data directory")
    parser.add_argument("--output-dir", type=str, default="../data/faiss_indexes", help="Output directory for indexes")
    parser.add_argument("--videos", nargs="*", help="Optional list of video IDs to process")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration for FAISS")
    
    args = parser.parse_args()
    
    build_and_save_indexes(
        data_root=args.data_root,
        output_dir=args.output_dir,
        videos=args.videos,
        use_gpu=args.use_gpu
    )
