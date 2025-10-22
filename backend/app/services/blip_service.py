import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
from typing import List, Optional
import os

logger = logging.getLogger(__name__)

class BLIPCaptionService:
    def __init__(self, use_fast: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.use_fast = use_fast
        self._load_model()
    
    def _load_model(self):
        """Load BLIP model for Vietnamese captioning"""
        try:
            logger.info("Loading BLIP model...")
            
            # Load BLIP model and processor
            model_name = "Salesforce/blip-image-captioning-large"
            logger.info(f"Loading model: {model_name}")
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name, dtype=torch.float32, device_map="auto").to(self.device)
            
            logger.info(f"BLIP model loaded successfully on {self.device}")
            logger.info(f"Model name: {model_name}")
            #logger.info(f"Model config: {self.model.config}")
            
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            raise
    
    def generate_caption(self, image_path: str, max_length: int = 200) -> Optional[str]:
        """
        Generate Vietnamese caption for an image using BLIP
        
        Args:
            image_path: Path to the image file
            max_length: Maximum length of generated caption
            
        Returns:
            Vietnamese caption or None if error
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # English prompt for BLIP (BLIP works better with English)
            english_prompt = "Describe this image in detail:"
            
            # Process inputs
            inputs = self.processor(
                images=image, 
                #text=english_prompt,
                return_tensors="pt"
            ).to(self.device, torch.float32)
            
            # Generate caption
            generation_params = {
                "max_length": max_length,
                "num_beams": 15,
               
                "do_sample": False,
                "use_cache": True
            }
            logger.debug(f"Generation params: {generation_params}")
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    **generation_params
                )
            
            # Decode caption
            raw_caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Raw caption: {raw_caption}")
            
            # Clean up caption (remove prompt if it appears)
            caption = raw_caption
            if english_prompt in caption:
                caption = caption.replace(english_prompt, "").strip()
                logger.info(f"Cleaned caption: {caption}")
            
            # Ensure it's not empty
            if not caption or caption.strip() == "":
                caption = "Không thể mô tả nội dung hình ảnh"
            
            logger.info(f"Generated caption for {image_path}: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded BLIP model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": "Salesforce/blip-image-captioning-large",
            "device": self.device,
            "model_type": type(self.model).__name__,
            "processor_type": type(self.processor).__name__,
            
        }
    
    def generate_captions_batch(self, image_paths: List[str], max_length: int = 50) -> List[Optional[str]]:
        """
        Generate captions for multiple images
        
        Args:
            image_paths: List of image paths
            max_length: Maximum length of generated captions
            
        Returns:
            List of captions (None for failed ones)
        """
        captions = []
        for image_path in image_paths:
            caption = self.generate_caption(image_path, max_length)
            captions.append(caption)
        return captions

# Global instance
blip_service = None

def get_blip_service(use_fast: bool = True) -> BLIPCaptionService:
    """Get or create BLIP service instance"""
    global blip_service
    if blip_service is None:
        blip_service = BLIPCaptionService(use_fast=use_fast)
    return blip_service
