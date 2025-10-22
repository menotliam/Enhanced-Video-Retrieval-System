/**
 * Vietnamese text normalization and highlighting utilities
 */
import React from 'react';

// Vietnamese accent mapping for normalization
const VIETNAMESE_ACCENTS: Record<string, string> = {
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
  'đ': 'd',
  'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
  'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
  'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
  'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
  'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
  'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
  'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
  'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
  'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
  'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
  'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
  'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
  'Đ': 'D'
};

/**
 * Remove Vietnamese accents from text
 * @param text - Text to normalize
 * @returns Normalized text without accents
 */
export const removeVietnameseAccents = (text: string): string => {
  return text.split('').map(char => VIETNAMESE_ACCENTS[char] || char).join('');
};

/**
 * Normalize Vietnamese text for search
 * @param text - Text to normalize
 * @returns Normalized text
 */
export const normalizeVietnameseText = (text: string): string => {
  return removeVietnameseAccents(text.toLowerCase().trim());
};

/**
 * Check if text contains search terms (case and accent insensitive)
 * @param text - Text to search in
 * @param searchTerms - Array of search terms
 * @returns Whether text contains any search terms
 */
export const containsSearchTerms = (text: string, searchTerms: string[]): boolean => {
  const normalizedText = normalizeVietnameseText(text);
  return searchTerms.some(term => 
    normalizedText.includes(normalizeVietnameseText(term))
  );
};

/**
 * Find all matches of search terms in text
 * @param text - Text to search in
 * @param searchTerms - Array of search terms
 * @returns Array of match objects with position and term
 */
export const findSearchMatches = (
  text: string, 
  searchTerms: string[]
): Array<{ start: number; end: number; term: string; originalTerm: string }> => {
  const matches: Array<{ start: number; end: number; term: string; originalTerm: string }> = [];
  const normalizedText = normalizeVietnameseText(text);
  
  searchTerms.forEach(originalTerm => {
    const normalizedTerm = normalizeVietnameseText(originalTerm);
    let startIndex = 0;
    
    while (true) {
      const index = normalizedText.indexOf(normalizedTerm, startIndex);
      if (index === -1) break;
      
      matches.push({
        start: index,
        end: index + normalizedTerm.length,
        term: normalizedTerm,
        originalTerm
      });
      
      startIndex = index + 1;
    }
  });
  
  // Sort matches by start position
  return matches.sort((a, b) => a.start - b.start);
};

/**
 * Highlight search terms in text with HTML markup
 * @param text - Original text
 * @param searchTerms - Array of search terms to highlight
 * @param highlightClass - CSS class for highlighting (default: 'highlight')
 * @returns Text with highlighted terms wrapped in HTML
 */
export const highlightVietnameseText = (
  text: string, 
  searchTerms: string[], 
  highlightClass: string = 'highlight'
): string => {
  if (!searchTerms.length || !text) return text;
  
  const matches = findSearchMatches(text, searchTerms);
  if (!matches.length) return text;
  
  // Create highlighted text by replacing matches
  let result = '';
  let lastIndex = 0;
  
  matches.forEach(match => {
    // Add text before match
    result += text.slice(lastIndex, match.start);
    
    // Add highlighted match
    const originalText = text.slice(match.start, match.end);
    result += `<span class="${highlightClass}">${originalText}</span>`;
    
    lastIndex = match.end;
  });
  
  // Add remaining text
  result += text.slice(lastIndex);
  
  return result;
};

/**
 * Highlight search terms in text with React JSX elements
 * @param text - Original text
 * @param searchTerms - Array of search terms to highlight
 * @param highlightComponent - React component for highlighting
 * @returns Array of text and highlight elements
 */
export const highlightVietnameseTextJSX = (
  text: string,
  searchTerms: string[],
  highlightComponent: React.ComponentType<{ children: React.ReactNode }>
): React.ReactNode[] => {
  if (!searchTerms.length || !text) return [text];
  
  const matches = findSearchMatches(text, searchTerms);
  if (!matches.length) return [text];
  
  const result: React.ReactNode[] = [];
  let lastIndex = 0;
  
  matches.forEach((match, index) => {
    // Add text before match
    if (match.start > lastIndex) {
      result.push(text.slice(lastIndex, match.start));
    }
    
    // Add highlighted match
    const originalText = text.slice(match.start, match.end);
    const HighlightComponent = highlightComponent;
    result.push(
      React.createElement(HighlightComponent, { key: `highlight-${index}`, children: originalText })
    );
    
    lastIndex = match.end;
  });
  
  // Add remaining text
  if (lastIndex < text.length) {
    result.push(text.slice(lastIndex));
  }
  
  return result;
};

/**
 * Calculate similarity score between two Vietnamese texts
 * @param text1 - First text
 * @param text2 - Second text
 * @returns Similarity score (0-1)
 */
export const calculateVietnameseSimilarity = (text1: string, text2: string): number => {
  const normalized1 = normalizeVietnameseText(text1);
  const normalized2 = normalizeVietnameseText(text2);
  
  if (normalized1 === normalized2) return 1;
  
  // Simple Levenshtein distance-based similarity
  const distance = levenshteinDistance(normalized1, normalized2);
  const maxLength = Math.max(normalized1.length, normalized2.length);
  
  return maxLength > 0 ? 1 - (distance / maxLength) : 0;
};

/**
 * Calculate Levenshtein distance between two strings
 * @param str1 - First string
 * @param str2 - Second string
 * @returns Levenshtein distance
 */
const levenshteinDistance = (str1: string, str2: string): number => {
  const matrix: number[][] = [];
  
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }
  
  return matrix[str2.length][str1.length];
};

/**
 * Extract keywords from Vietnamese text
 * @param text - Text to extract keywords from
 * @param maxKeywords - Maximum number of keywords to extract
 * @returns Array of keywords
 */
export const extractVietnameseKeywords = (text: string, maxKeywords: number = 10): string[] => {
  // Vietnamese stop words
  const stopWords = new Set([
    'và', 'của', 'cho', 'với', 'từ', 'trong', 'ngoài', 'trên', 'dưới', 'trước', 'sau',
    'có', 'không', 'là', 'được', 'bị', 'đã', 'sẽ', 'đang', 'vẫn', 'còn', 'chỉ', 'mới',
    'cũng', 'thì', 'mà', 'nếu', 'vì', 'do', 'bởi', 'tại', 'ở', 'tại', 'về', 'theo',
    'như', 'giống', 'khác', 'cùng', 'riêng', 'chung', 'tất', 'cả', 'mỗi', 'mọi', 'bất',
    'kỳ', 'nào', 'đâu', 'đây', 'đó', 'kia', 'này', 'ấy', 'vậy', 'thế', 'sao', 'tại sao',
    'làm sao', 'bao giờ', 'khi nào', 'ở đâu', 'từ đâu', 'đến đâu', 'về đâu', 'theo đâu'
  ]);
  
  const normalizedText = normalizeVietnameseText(text);
  const words = normalizedText.split(/\s+/).filter(word => 
    word.length > 1 && !stopWords.has(word)
  );
  
  // Count word frequency
  const wordCount: Record<string, number> = {};
  words.forEach(word => {
    wordCount[word] = (wordCount[word] || 0) + 1;
  });
  
  // Sort by frequency and return top keywords
  return Object.entries(wordCount)
    .sort(([,a], [,b]) => b - a)
    .slice(0, maxKeywords)
    .map(([word]) => word);
};

/**
 * Create search suggestions based on Vietnamese text
 * @param text - Base text for suggestions
 * @param suggestions - Array of possible suggestions
 * @param maxSuggestions - Maximum number of suggestions
 * @returns Array of relevant suggestions
 */
export const createVietnameseSuggestions = (
  text: string,
  suggestions: string[],
  maxSuggestions: number = 5
): string[] => {
  if (!text.trim()) return suggestions.slice(0, maxSuggestions);
  
  const normalizedText = normalizeVietnameseText(text);
  const textWords = normalizedText.split(/\s+/);
  
  // Score suggestions based on word overlap
  const scoredSuggestions = suggestions.map(suggestion => {
    const normalizedSuggestion = normalizeVietnameseText(suggestion);
    const suggestionWords = normalizedSuggestion.split(/\s+/);
    
    let score = 0;
    textWords.forEach(textWord => {
      suggestionWords.forEach(suggestionWord => {
        if (suggestionWord.includes(textWord) || textWord.includes(suggestionWord)) {
          score += 1;
        }
      });
    });
    
    return { suggestion, score };
  });
  
  // Sort by score and return top suggestions
  return scoredSuggestions
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxSuggestions)
    .map(item => item.suggestion);
};

/**
 * Check if two Vietnamese texts are similar (fuzzy matching)
 * @param text1 - First text
 * @param text2 - Second text
 * @param threshold - Similarity threshold (0-1, default: 0.8)
 * @returns Whether texts are similar
 */
export const areVietnameseTextsSimilar = (
  text1: string, 
  text2: string, 
  threshold: number = 0.8
): boolean => {
  return calculateVietnameseSimilarity(text1, text2) >= threshold;
};

/**
 * Get Vietnamese text statistics
 * @param text - Text to analyze
 * @returns Text statistics object
 */
export const getVietnameseTextStats = (text: string) => {
  const normalizedText = normalizeVietnameseText(text);
  const words = normalizedText.split(/\s+/).filter(word => word.length > 0);
  const characters = text.length;
  const charactersNoSpaces = text.replace(/\s/g, '').length;
  const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
  
  return {
    wordCount: words.length,
    characterCount: characters,
    characterCountNoSpaces: charactersNoSpaces,
    sentenceCount: sentences.length,
    averageWordLength: words.length > 0 ? charactersNoSpaces / words.length : 0,
    averageSentenceLength: sentences.length > 0 ? words.length / sentences.length : 0
  };
};
