"""
OpenAI LLM service for query parsing and RAG functionality
"""
import openai
from typing import Dict, Any, List, Optional
from loguru import logger

from app.config import settings


class LLMService:
    """OpenAI LLM service for Vietnamese query processing and RAG"""
    
    def __init__(self):
        self.client = None
        self.api_key = None
        self.use_llm = settings.USE_LLM
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.use_llm:
            logger.info("LLM features disabled - using offline mode")
            return
            
        try:
            # Get API key from environment or config
            self.api_key = settings.OPENAI_API_KEY
            
            if self.api_key:
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found, LLM features will be disabled")
                self.use_llm = False
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
            self.use_llm = False
    
    async def parse_and_expand_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse and expand Vietnamese query using OpenAI or rule-based fallback
        
        Args:
            query: Original Vietnamese query
            filters: Existing filters
            
        Returns:
            Parsed query with extracted filters and expanded terms
        """
        if not self.use_llm or not self.client:
            logger.info("Using rule-based query parsing (offline mode)")
            return self._rule_based_query_parsing(query, filters)
        
        try:
            system_prompt = """
            Bạn là một trợ lý AI chuyên xử lý truy vấn tìm kiếm video tiếng Việt và là hệ thống chuẩn hóa truy vấn cho CLIP retrieval.
            Nhiệm vụ thứ nhất của bạn là phân tích truy vấn và trích xuất thông tin có cấu trúc.
            Nhiệm vụ thứ hai của bạn là chuẩn hóa truy vấn cho CLIP retrieval và trả về trường english_query: từ query tiếng Việt, hãy tạo ra một câu mô tả ngắn gọn bằng tiếng Anh, giống như caption ảnh, tập trung vào: **đối tượng (chỉ trả về khái quát các loại đối tượng không cần cụ thể số lượng, màu sắc)**, hành động, bối cảnh trực quan và đầu ra của nhiệm vụ thứ hai này chỉ là 1 câu tiếng Anh cho trường english_query.
            
            Hãy phân tích truy vấn và trả về JSON với các trường sau:
            - original_query: truy vấn gốc
            - normalized_query: truy vấn đã chuẩn hóa tiếng Việt
            - english_query: truy vấn gốc Tiếng Việt được dịch sang tiếng Anh loại bỏ thông tin về thời gian, năm, mùa, tháng, ngày, giờ, phút, giây, địa điểm, sự kiện, cảm xúc. Ví dụ "Video of a charity gift-giving event taking place at a hospital during the Spring of 2024. In the scene, there are two men (in pink shirts and white shirts) standing on either side, representing the organizing committee. In the middle are four children and teenagers, including one in a red shirt, one in a white shirt, one in a pink skirt, and one in a blue shirt." thì trả về "a group of people including men and children holding prizes.". 
            - keywords: danh sách từ khóa quan trọng, hãy trả về keywords tiếng Việt nếu trong câu truy vấn ban đầu có keywords tiếng Anh thì hãy giữ tiếng Anh.
            - query_type: loại truy vấn (visual/audio/text)
            - extracted_filters: các bộ lọc được trích xuất, hãy trả về filters tiếng Việt nếu trong câu truy vấn ban đầu có filters tiếng Anh thì hãy giữ tiếng Anh.
            - expanded_terms: các từ khóa mở rộng liên quan, nếu có những từ về các biển báo như "biển cảnh báo" thì hãy tách ra thành "biển" và "cảnh báo". Chỉ trả về expanded_terms tiếng Việt.
            - confidence: độ tin cậy của việc phân tích (0-1)
            
            Ví dụ truy vấn: "Tìm video có người đi bộ trên đường phố Hà Nội vào buổi sáng"
            """
            
            user_prompt = f"""
            Truy vấn: "{query}"
            
            Hãy phân tích và trả về kết quả dưới dạng JSON.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed_result = self._parse_llm_response(content)
            
            # Merge with existing filters
            if filters:
                parsed_result["extracted_filters"].update(filters)
            
            logger.info(f"LLM parsed query: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM query parsing failed: {str(e)}")
            return self._rule_based_query_parsing(query, filters)
    
    def _rule_based_query_parsing(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback rule-based query parsing"""
        from app.utils.text_normalizer import text_normalizer
        
        normalized_query = text_normalizer.normalize_text(query)
        keywords = text_normalizer.extract_keywords(query)
        
        # Simple rule-based filter extraction
        extracted_filters = {}
        query_lower = query.lower()
        
        # Location extraction
        locations = ["hà nội", "sài gòn", "hồ chí minh", "tp.hcm", "hcm", "đà nẵng", "huế"]
        for location in locations:
            if location in query_lower:
                extracted_filters["location"] = location
                break
        
        # Time extraction
        time_patterns = {"sáng": "morning", "trưa": "noon", "chiều": "afternoon", "tối": "evening", "đêm": "night"}
        for vn_time, en_time in time_patterns.items():
            if vn_time in query_lower:
                extracted_filters["time_of_day"] = en_time
                break
        
        # Object extraction
        objects = ["người", "xe", "cây", "nhà", "đường", "cửa hàng", "công viên"]
        found_objects = [obj for obj in objects if obj in query_lower]
        if found_objects:
            extracted_filters["objects"] = found_objects
        
        # Query type detection
        if any(word in query_lower for word in ["hình", "ảnh", "video", "cảnh"]):
            query_type = "visual"
        elif any(word in query_lower for word in ["tiếng", "âm thanh", "nói", "hát"]):
            query_type = "audio"
        else:
            query_type = "text"
        
        result = {
            "original_query": query,
            "normalized_query": normalized_query,
            "keywords": keywords,
            "query_type": query_type,
            "extracted_filters": extracted_filters,
            "expanded_terms": keywords,  # Simple expansion
            "confidence": 0.7
        }
        
        # Merge with existing filters
        if filters:
            result["extracted_filters"].update(filters)
        
        return result
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON"""
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            # Return default structure
            return {
                "original_query": "",
                "normalized_query": "",
                "keywords": [],
                "query_type": "text",
                "extracted_filters": {},
                "expanded_terms": [],
                "confidence": 0.0
            }
    
    async def generate_rag_answer(self, scenes: List[Dict[str, Any]], query: str) -> Optional[str]:
        """
        Generate RAG answer using OpenAI or return None if disabled
        
        Args:
            scenes: List of relevant scenes
            query: Original query
            
        Returns:
            Generated answer in Vietnamese or None if LLM disabled
        """
        if not self.use_llm or not self.client or not scenes:
            logger.info("RAG disabled - returning None")
            return None
        
        try:
            # Prepare context from scenes
            context = self._prepare_rag_context(scenes)
            
            system_prompt = """
            Bạn là một trợ lý AI chuyên tìm kiếm video. Dựa trên thông tin từ các cảnh video, 
            hãy trả lời câu hỏi của người dùng một cách ngắn gọn và chính xác bằng tiếng Việt.
            """
            
            user_prompt = f"""
            Câu hỏi: "{query}"
            
            Thông tin từ các cảnh video:
            {context}
            
            Hãy trả lời câu hỏi dựa trên thông tin trên.
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated RAG answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"RAG answer generation failed: {str(e)}")
            return None
    
    def _prepare_rag_context(self, scenes: List[Dict[str, Any]]) -> str:
        """Prepare context for RAG from scenes with BLIP captions"""
        from .blip_service import get_blip_service
        
        context_parts = []
        blip_service = get_blip_service(use_fast=settings.BLIP_USE_FAST)
        
        for i, scene in enumerate(scenes[:5], 1):  # Limit to top 5 scenes
            scene_info = f"Cảnh {i}: "
            
            # Generate BLIP caption if image path available
            image_path = scene.get("image_path")
            blip_caption = None
            if image_path:
                try:
                    blip_caption = blip_service.generate_caption(image_path)
                except Exception as e:
                    logger.warning(f"Failed to generate BLIP caption for {image_path}: {e}")
            
            # Build context with all available information
            if blip_caption:
                scene_info += f"BLIP Caption: {blip_caption} | "
            
            if scene.get("asr_text"):
                scene_info += f"ASR: {scene['asr_text']} | "
            if scene.get("ocr_texts"):
                ocr_text = " ".join(scene['ocr_texts']) if isinstance(scene['ocr_texts'], list) else scene['ocr_texts']
                scene_info += f"OCR: {ocr_text} | "
            if scene.get("scene_label"):
                scene_info += f"Scene: {scene['scene_label']} | "
            
            # Remove trailing " | " if exists
            scene_info = scene_info.rstrip(" | ")
            context_parts.append(scene_info)
        
        return "\n".join(context_parts)
    
    async def correct_vietnamese_ocr(self, ocr_text: str) -> str:
        """
        Correct Vietnamese OCR text by restoring proper diacritics
        
        Args:
            ocr_text: Raw OCR text without proper diacritics
            
        Returns:
            Corrected text with proper Vietnamese diacritics
        """
        if not ocr_text or not ocr_text.strip():
            return ocr_text
            
        # If LLM not available, try rule-based correction
        if not self.use_llm or not self.client:
            logger.info("Using rule-based Vietnamese text correction")
            return self._rule_based_vietnamese_correction(ocr_text)
        
        try:
            system_prompt = """
            Bạn là một trợ lý AI chuyên sửa lỗi chính tả tiếng Việt. Nhiệm vụ của bạn là khôi phục 
            dấu thanh điệu tiếng Việt cho văn bản OCR đã mất dấu.
            
            Nguyên tắc:
            - Chỉ thêm dấu thanh điệu, không thay đổi nội dung
            - Nếu dấu thanh điệu nó bất thường thì hãy sử dụng ngữ cảnh để xác định từ đúng
            - Nếu không chắc chắn, giữ nguyên từ gốc
            - Trả về văn bản đã được sửa, không thêm giải thích
            
            Ví dụ:
            - "SUT LÜN ö" → "Sụt lún ở"
            - "dong song" → "dòng sông"
            - "nha hang" → "nhà hàng"  
            - "xe may" → "xe máy"
            """
            
            user_prompt = f"""
            Văn bản OCR cần sửa dấu: "{ocr_text}"
            
            Hãy sửa lại văn bản này với dấu thanh điệu tiếng Việt chính xác:
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            corrected_text = response.choices[0].message.content.strip()
            
            # Remove quotes if LLM added them
            if corrected_text.startswith('"') and corrected_text.endswith('"'):
                corrected_text = corrected_text[1:-1]
            
            logger.info(f"OCR correction: '{ocr_text}' → '{corrected_text}'")
            return corrected_text
            
        except Exception as e:
            logger.error(f"LLM OCR correction failed: {str(e)}")
            return self._rule_based_vietnamese_correction(ocr_text)
    
    def _rule_based_vietnamese_correction(self, text: str) -> str:
        """
        Rule-based Vietnamese diacritic restoration (fallback)
        """
        # Common Vietnamese word corrections
        corrections = {
            "dong song": "dòng sông",
            "nha hang": "nhà hàng",
            "xe may": "xe máy",
            "ban an": "bàn ăn",
            "cua hang": "cửa hàng",
            "truong hoc": "trường học",
            "benh vien": "bệnh viện",
            "cong vien": "công viên",
            "nha tho": "nhà thờ",
            "cho ca": "chợ cá",
            "duong pho": "đường phố",
            "quan ca phe": "quán cà phê",
            "nha tram": "nhà trạm",
            "cau duong": "cầu đường",
            "truyen hinh": "truyền hình",
            "dien thoai": "điện thoại",
            "ban do": "bản đồ",
            "xe buyt": "xe buýt",
            "nha ga": "nhà ga",
            "san bay": "sân bay",
            "cho ben": "chợ Bến",
            "sanh": "sảnh",
            "lau": "lầu",
            "tang": "tầng",
            "cong ty": "công ty",
            "van phong": "văn phòng"
        }
        
        corrected = text
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        logger.info(f"Rule-based correction: '{text}' → '{corrected}'")
        return corrected

    async def generate_answer_with_context(self, query: str, context: str) -> Optional[str]:
        """
        Generate answer using context and query
        
        Args:
            query: User question
            context: Context information from RAG pipeline
            
        Returns:
            Generated answer or None if error
        """
        if not self.use_llm or not self.client:
            logger.warning("LLM service not available, cannot generate answer")
            return None
        
        try:
            system_prompt = """
            Bạn là một trợ lý AI chuyên trả lời câu hỏi dựa trên thông tin được cung cấp.
            Nhiệm vụ của bạn là phân tích thông tin context và trả lời câu hỏi một cách chính xác và ngắn gọn.
            
            Nguyên tắc:
            - Các thông tin trong context có thể sẽ không liên quan đến nhau
            - Nếu thấy bất kì thông tin nào trong context chưa chuẩn Tiếng Việt thì hãy chuẩn hóa tiếng Việt đảm bảo độ chính xác và dùng nó để làm context trả lời câu hỏi
            - Chỉ trả lời dựa trên thông tin có trong context
            - Nếu không có thông tin đủ để trả lời, hãy nói rõ
            - Trả lời bằng tiếng Việt, ngắn gọn và chính xác
            - Không thêm thông tin không có trong context
            """
            
            user_prompt = f"""
            Context thông tin:
            {context}
            
            Câu hỏi: {query}
            
            Hãy trả lời câu hỏi dựa trên thông tin context trên:
            """
            
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for query '{query}': {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with context: {str(e)}")
            return None

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.use_llm and self.client is not None


# Global LLM service instance
llm_service = LLMService()
