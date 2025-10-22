import axios, { AxiosResponse } from 'axios';
import { SearchResult, SearchQuery, FilterOptions, SystemStats } from '@/types';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for complex searches
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response?.status === 429) {
      throw new Error('Quá nhiều yêu cầu. Vui lòng thử lại sau.');
    } else if (error.response?.status >= 500) {
      throw new Error('Lỗi máy chủ. Vui lòng thử lại sau.');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Hết thời gian chờ. Vui lòng thử lại.');
    }
    
    return Promise.reject(error);
  }
);

// Text Search API
export const searchByText = async (
  query: string,
  top_k: number = 20,
  use_rag: boolean = false,
  filters?: Record<string, any>
): Promise<SearchResult> => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', top_k.toString());
  formData.append('use_rag', use_rag.toString());
  
  if (filters) {
    formData.append('filters', JSON.stringify(filters));
  }

  const response: AxiosResponse<SearchResult> = await api.post('/search/text', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// Image Search API
export const searchByImage = async (
  imageFile: File,
  query?: string,
  top_k: number = 20,
  use_rag: boolean = false,
  filters?: Record<string, any>
): Promise<SearchResult> => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('top_k', top_k.toString());
  formData.append('use_rag', use_rag.toString());
  
  if (query) {
    formData.append('query', query);
  }
  
  if (filters) {
    formData.append('filters', JSON.stringify(filters));
  }

  const response: AxiosResponse<SearchResult> = await api.post('/search/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// Audio Search API
export const searchByAudio = async (
  audioFile: File,
  query?: string,
  top_k: number = 20,
  use_rag: boolean = false,
  filters?: Record<string, any>
): Promise<SearchResult> => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('top_k', top_k.toString());
  formData.append('use_rag', use_rag.toString());
  
  if (query) {
    formData.append('query', query);
  }
  
  if (filters) {
    formData.append('filters', JSON.stringify(filters));
  }

  const response: AxiosResponse<SearchResult> = await api.post('/search/audio', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// Generic Search Function
export const performSearch = async (searchQuery: SearchQuery): Promise<SearchResult> => {
  const { query, type, image, audio, filters, top_k = 20, use_rag = false } = searchQuery;

  switch (type) {
    case 'text':
      return searchByText(query, top_k, use_rag, filters);
    
    case 'image':
      if (!image) {
        throw new Error('Cần có ảnh để tìm kiếm bằng hình ảnh');
      }
      return searchByImage(image, query, top_k, use_rag, filters);
    
    case 'audio':
      if (!audio) {
        throw new Error('Cần có tệp âm thanh để tìm kiếm bằng giọng nói');
      }
      return searchByAudio(audio, query, top_k, use_rag, filters);
    
    default:
      throw new Error(`Loại tìm kiếm không hỗ trợ: ${type}`);
  }
};

// Get Search Suggestions
export const getSearchSuggestions = async (
  query: string,
  limit: number = 10
): Promise<{ query: string; suggestions: string[] }> => {
  const response = await api.get('/search/frames-cli', {
    params: { query, limit },
  });

  return response.data;
};

// Get Filter Options
export const getFilterOptions = async (): Promise<FilterOptions> => {
  const response = await api.get('/search/filters');
  return response.data;
};

// Get System Statistics
export const getSystemStats = async (): Promise<SystemStats> => {
  const response = await api.get('/search/statistics');
  return response.data;
};

// Health Check
export const checkHealth = async (): Promise<{ status: string; components: Record<string, boolean> }> => {
  const response = await api.post('/search/health');
  return response.data;
};

// Performance Monitoring
export interface PerformanceMetrics {
  searchTime: number;
  apiLatency: number;
  totalTime: number;
}

export const measurePerformance = async <T>(
  apiCall: () => Promise<T>
): Promise<{ result: T; metrics: PerformanceMetrics }> => {
  const startTime = performance.now();
  
  try {
    const result = await apiCall();
    const endTime = performance.now();
    
    const metrics: PerformanceMetrics = {
      searchTime: endTime - startTime,
      apiLatency: endTime - startTime,
      totalTime: endTime - startTime,
    };

    return { result, metrics };
  } catch (error) {
    const endTime = performance.now();
    console.error('Performance measurement failed:', {
      duration: endTime - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
    throw error;
  }
};

// Cache Management
const searchCache = new Map<string, { data: SearchResult; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const getCachedSearch = (cacheKey: string): SearchResult | null => {
  const cached = searchCache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    console.log('🎯 Cache hit for:', cacheKey);
    return cached.data;
  }
  
  if (cached) {
    searchCache.delete(cacheKey);
  }
  
  return null;
};

export const setCachedSearch = (cacheKey: string, data: SearchResult): void => {
  searchCache.set(cacheKey, {
    data,
    timestamp: Date.now(),
  });
  
  // Clean old cache entries
  if (searchCache.size > 50) {
    const oldestKey = [...searchCache.keys()][0];
    searchCache.delete(oldestKey);
  }
};

export const clearSearchCache = (): void => {
  searchCache.clear();
  console.log('🗑️ Search cache cleared');
};

// Search with caching
export const performSearchWithCache = async (
  searchQuery: SearchQuery
): Promise<SearchResult> => {
  // Create cache key from search parameters
  const cacheKey = JSON.stringify({
    query: searchQuery.query,
    type: searchQuery.type,
    filters: searchQuery.filters,
    top_k: searchQuery.top_k,
    // Don't include files in cache key
  });

  // Try to get from cache first (only for text searches)
  if (searchQuery.type === 'text') {
    const cached = getCachedSearch(cacheKey);
    if (cached) {
      return cached;
    }
  }

  // Perform actual search
  const result = await performSearch(searchQuery);

  // Cache the result (only for text searches)
  if (searchQuery.type === 'text') {
    setCachedSearch(cacheKey, result);
  }

  return result;
};

// Simple Frames CLI search endpoint
export interface FrameCliItem {
  video_id: string;
  frame_idx: number;
  score: number;
  image_url?: string | null;
  image?: string | null;
  youtube_url?: string | null;
  frame_timestamp?: number | null;
}

export interface FramesCliResponse {
  query: string;
  total_results: number;
  results: FrameCliItem[];
  search_time: number;
}

export const searchFramesCLI = async (
  query: string,
  top_k: number
): Promise<FramesCliResponse> => {
  const form = new FormData();
  form.append('query', query);
  form.append('top_k', String(top_k));
  const resp = await api.post('/search/frames-cli', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return resp.data as FramesCliResponse;
};

// RAG API
export interface RAGResponse {
  answer: string;
  frame_info: {
    video_id: string;
    frame_idx: number;
    image_url: string;
  };
  processing_time: number;
}

export const generateRAGAnswer = async (
  frameInput: string,
  query: string
): Promise<RAGResponse> => {
  const form = new FormData();
  form.append('frame_input', frameInput);
  form.append('query', query);
  
  const response = await api.post('/search/rag/generate', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  
  return response.data as RAGResponse;
};

// CSV Export API
export interface CSVExportOptions {
  query: string;
  top_k?: number;
  pre_filter?: boolean;
  pre_filter_limit?: number;
  filename?: string;
}

export const exportFramesToCSV = async (options: CSVExportOptions): Promise<Blob> => {
  console.log('🔍 CSV Export Options:', options);  // Debug log
  
  const form = new FormData();
  form.append('query', options.query);
  form.append('top_k', String(options.top_k ?? 10));  // Use nullish coalescing instead of ||
  form.append('pre_filter', String(options.pre_filter || false));
  
  if (options.pre_filter_limit) {
    form.append('pre_filter_limit', String(options.pre_filter_limit));
  }
  
  if (options.filename) {
    form.append('filename', options.filename);
  }

  const response = await api.post('/search/frames-cli/export-csv', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob', // Quan trọng để nhận file
  });

  return response.data;
};

// Helper function to download CSV file
export const downloadCSV = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

// Combined function to search and export CSV
export const searchAndExportCSV = async (options: CSVExportOptions): Promise<void> => {
  try {
    const blob = await exportFramesToCSV(options);
    const filename = options.filename || `search_results_${Date.now()}.csv`;
    downloadCSV(blob, filename);
  } catch (error) {
    console.error('CSV export failed:', error);
    throw error;
  }
};