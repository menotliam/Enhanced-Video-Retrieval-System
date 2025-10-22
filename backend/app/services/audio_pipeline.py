"""
Audio pipeline: cut audio, ASR, text normalization, audio & transcript embedding.
Input: Scene audio segment
Output: Dict with transcript, embeddings
"""
from typing import Dict

class AudioPipeline:
    def __init__(self, asr_model, text_normalizer, audio_embedding_model, transcript_embedding_model):
        pass

    def process_scene_audio(self, audio_path: str, scene_id: str, start_ts: str, end_ts: str) -> Dict:
        """
        Process audio for a scene: cut, ASR, normalize, embed.
        Returns: Dict (see spec)
        """
        pass
