"""
Vietnamese text normalization utilities
"""
import re
import unicodedata
from typing import List, Optional
from underthesea import word_tokenize
from vncorenlp import VnCoreNLP

class VietnameseTextNormalizer:
    """Vietnamese text normalization and processing"""
    
    def __init__(self, remove_tones: bool = False, lowercase: bool = True):
        self.remove_tones = remove_tones
        self.lowercase = lowercase
        
        # Vietnamese tone marks mapping
        self.tone_marks = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        # Common Vietnamese abbreviations and slang
        self.abbreviations = {
            'vn': 'việt nam',
            'vnese': 'việt nam',
            'viet': 'việt',
            'sg': 'sài gòn',
            'hn': 'hà nội',
            'hcm': 'hồ chí minh',
            'tp': 'thành phố',
            'q': 'quận',
            'p': 'phường',
            'f': 'phường',
            'st': 'số',
            'đt': 'điện thoại',
            'dt': 'điện thoại',
            'fb': 'facebook',
            'ig': 'instagram',
            'yt': 'youtube',
            'ytb': 'youtube',
            'vs': 'với',
            'vk': 'với',
            'ko': 'không',
            'k': 'không',
            'kg': 'không',
            'dc': 'được',
            'đc': 'được',
            'mk': 'mình',
            'mình': 'mình',
            'bn': 'bao nhiêu',
            'bnhiêu': 'bao nhiêu',
            'bh': 'bao giờ',
            'bgiờ': 'bao giờ',
            'lm': 'làm',
            'làm': 'làm',
            'j': 'gì',
            'gì': 'gì',
            's': 'sao',
            'sao': 'sao',
            'z': 'vậy',
            'vậy': 'vậy',
            'zậy': 'vậy',
            'ok': 'ok',
            'okie': 'ok',
            'oki': 'ok'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text"""
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Basic normalization
        text = text.strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle abbreviations
        text = self._expand_abbreviations(text)
        
        # Normalize using underthesea
        try:
            text = text_normalize(text)
        except:
            pass  # Fallback if normalization fails
        
        # Remove tones if requested
        if self.remove_tones:
            text = self._remove_tone_marks(text)
        
        # Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand Vietnamese abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for abbreviation lookup
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.abbreviations:
                # Preserve original punctuation
                punctuation = re.sub(r'[\w]', '', word)
                expanded_words.append(self.abbreviations[clean_word] + punctuation)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _remove_tone_marks(self, text: str) -> str:
        """Remove Vietnamese tone marks"""
        for tone_mark, base in self.tone_marks.items():
            text = text.replace(tone_mark, base)
            text = text.replace(tone_mark.upper(), base.upper())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        normalized_text = self.normalize_text(text)
        
        try:
            # Use underthesea for Vietnamese tokenization
            tokens = word_tokenize(normalized_text)
            return [token for token in tokens if token.strip()]
        except:
            # Fallback to simple tokenization
            return normalized_text.split()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text"""
        tokens = self.tokenize(text)
        
        # Remove stop words and short tokens
        stop_words = {
            'và', 'của', 'cho', 'với', 'từ', 'trong', 'ngoài', 'trên', 'dưới',
            'trước', 'sau', 'giữa', 'bên', 'cạnh', 'gần', 'xa', 'lớn', 'nhỏ',
            'cao', 'thấp', 'dài', 'ngắn', 'rộng', 'hẹp', 'đẹp', 'xấu', 'tốt',
            'xấu', 'mới', 'cũ', 'nhanh', 'chậm', 'nhiều', 'ít', 'tất', 'cả',
            'mỗi', 'mọi', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy',
            'tám', 'chín', 'mười', 'trăm', 'nghìn', 'triệu', 'tỷ'
        }
        
        keywords = []
        for token in tokens:
            if (len(token) > 1 and 
                token.lower() not in stop_words and
                not token.isdigit() and
                not re.match(r'^[^\w\s]+$', token)):
                keywords.append(token)
        
        return keywords[:max_keywords]
    
    def fuzzy_match(self, query: str, target: str, threshold: float = 0.8) -> bool:
        """Fuzzy string matching for Vietnamese text"""
        from difflib import SequenceMatcher
        
        normalized_query = self.normalize_text(query)
        normalized_target = self.normalize_text(target)
        
        similarity = SequenceMatcher(None, normalized_query, normalized_target).ratio()
        return similarity >= threshold
    
    def search_preprocessing(self, text: str) -> str:
        """Preprocess text for search queries"""
        # Normalize and tokenize
        tokens = self.tokenize(text)
        
        # Remove very short tokens and common words
        filtered_tokens = []
        for token in tokens:
            if len(token) > 2 and not token.isdigit():
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)

# Global instance
text_normalizer = VietnameseTextNormalizer()
