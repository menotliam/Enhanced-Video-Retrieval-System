"""
Vietnamese Cross-Encoder model for re-ranking search results
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModel
from loguru import logger

from app.config import settings, MODEL_CONFIG


class VietnameseCrossEncoder:
    """Vietnamese-specific cross-encoder for re-ranking search results"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        
        # Use PhoBERT as base model for cross-encoder
        self.model_name = "vinai/phobert-base"
        
    def load_model(self):
        """Load Vietnamese cross-encoder model"""
        logger.info("Loading Vietnamese cross-encoder model...")
        
        try:
            # Load PhoBERT tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Vietnamese cross-encoder model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def encode_query_document_pair(
        self, 
        query: str, 
        document: str
    ) -> float:
        """
        Encode query-document pair and return similarity score
        
        Args:
            query: Vietnamese query text
            document: Vietnamese document text (scene transcript, OCR, etc.)
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.model or not self.tokenizer:
            logger.warning("Cross-encoder model not available")
            return 0.5  # Default score
        
        try:
            # Prepare input
            input_text = f"{query} [SEP] {document}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding for similarity
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # Calculate similarity score (simple approach)
                # In a real implementation, you would use a trained cross-encoder head
                similarity_score = self._calculate_similarity_score(cls_embedding)
                
                return similarity_score.item()
                
        except Exception as e:
            logger.error(f"Cross-encoder encoding failed: {str(e)}")
            return 0.5  # Default score
    
    def _calculate_similarity_score(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity score from embedding
        This is a simplified approach - in production, use a trained cross-encoder head
        """
        # Simple approach: use the norm of the embedding as a proxy for relevance
        # Higher norm = more relevant
        norm = torch.norm(embedding, dim=1)
        
        # Normalize to [0, 1] range using sigmoid
        score = torch.sigmoid(norm / 10.0)  # Scale factor for normalization
        
        return score
    
    def batch_encode_pairs(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[float]:
        """
        Encode multiple query-document pairs in batch
        
        Args:
            query: Vietnamese query text
            documents: List of Vietnamese document texts
            
        Returns:
            List of similarity scores
        """
        if not self.model or not self.tokenizer:
            logger.warning("Cross-encoder model not available")
            return [0.5] * len(documents)  # Default scores
        
        try:
            scores = []
            
            # Process in batches to avoid memory issues
            batch_size = 8
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Prepare batch inputs
                batch_texts = [f"{query} [SEP] {doc}" for doc in batch_docs]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    # Calculate scores
                    batch_scores = self._calculate_similarity_score(cls_embeddings)
                    scores.extend(batch_scores.cpu().numpy().tolist())
            
            return scores
            
        except Exception as e:
            logger.error(f"Batch cross-encoder encoding failed: {str(e)}")
            return [0.5] * len(documents)  # Default scores
    
    def re_rank_scenes(
        self, 
        query: str, 
        scenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank scenes using cross-encoder
        
        Args:
            query: Vietnamese query text
            scenes: List of scene dictionaries
            
        Returns:
            Re-ranked scenes with cross-encoder scores
        """
        if not scenes:
            return []
        
        try:
            # Prepare documents for cross-encoder
            documents = []
            for scene in scenes:
                # Combine scene text information
                scene_text = ""
                if scene.get("transcript"):
                    scene_text += scene["transcript"] + " "
                if scene.get("ocr_text"):
                    scene_text += scene["ocr_text"] + " "
                if scene.get("scene_description"):
                    scene_text += scene["scene_description"] + " "
                
                documents.append(scene_text.strip())
            
            # Get cross-encoder scores
            cross_encoder_scores = self.batch_encode_pairs(query, documents)
            
            # Add scores to scenes
            for i, scene in enumerate(scenes):
                scene["cross_encoder_score"] = cross_encoder_scores[i]
            
            # Sort by cross-encoder score
            scenes.sort(key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
            
            logger.info(f"Re-ranked {len(scenes)} scenes using cross-encoder")
            return scenes
            
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {str(e)}")
            return scenes  # Return original order
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the cross-encoder model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "loaded": self.model is not None
        }


# Global cross-encoder instance
cross_encoder = VietnameseCrossEncoder()
