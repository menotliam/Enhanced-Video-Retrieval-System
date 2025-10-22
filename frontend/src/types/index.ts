// API Response Types
export interface SearchResult {
  query: string;
  total_results: number;
  scenes: Scene[];
  search_time: number;
  rag_answer?: string;
}

export interface Scene {
  scene_id: string;
  video_id: string;
  start_time: number;
  end_time: number;
  duration: number;
  score: number;
  transcript: string;
  ocr_text: string;
  detected_objects: string[];
  scene_description: string;
  preview_image_url: string;
  audio_snippet_url: string;
  metadata: SceneMetadata;
}

export interface SceneMetadata {
  location?: string;
  time_of_day?: string;
  weather?: string;
  cross_encoder_score: number;
  fuzzy_score: number;
  quality_score: number;
}

// Search Types
export type SearchType = 'text' | 'image' | 'audio';

export interface SearchQuery {
  query: string;
  type: SearchType;
  image?: File;
  audio?: File;
  filters?: SearchFilters;
  top_k?: number;
  use_rag?: boolean;
}

export interface SearchFilters {
  objects?: string[];
  locations?: string[];
  time_of_day?: string[];
  weather?: string[];
  date_range?: {
    start: Date;
    end: Date;
  };
}

// UI State Types
export interface UIState {
  isLoading: boolean;
  error: string | null;
  searchHistory: SearchHistoryItem[];
  selectedScene: Scene | null;
  viewMode: 'grid' | 'list' | 'timeline';
  sortBy: 'relevance' | 'time' | 'quality';
  filterPanelOpen: boolean;
}

export interface SearchHistoryItem {
  id: string;
  query: string;
  type: SearchType;
  timestamp: Date;
  resultCount: number;
}

// Filter Options
export interface FilterOptions {
  objects: string[];
  locations: string[];
  time_of_day: string[];
  weather: string[];
}

// Component Props
export interface SearchBarProps {
  onSearch: (query: SearchQuery) => void;
  isLoading: boolean;
  placeholder?: string;
}

export interface SceneCardProps {
  scene: Scene;
  onClick: (scene: Scene) => void;
  viewMode: 'grid' | 'list';
  highlightTerms?: string[];
}

export interface VideoPlayerProps {
  scene: Scene;
  autoPlay?: boolean;
  onClose: () => void;
}

export interface FiltersProps {
  filters: SearchFilters;
  options: FilterOptions;
  onFiltersChange: (filters: SearchFilters) => void;
  isOpen: boolean;
  onToggle: () => void;
}

// API Error Types
export interface APIError {
  message: string;
  status: number;
  details?: any;
}

// Audio Recording Types
export interface AudioRecording {
  blob: Blob;
  duration: number;
  url: string;
}

// Image Upload Types
export interface ImageUpload {
  file: File;
  preview: string;
  dimensions?: {
    width: number;
    height: number;
  };
}

// Statistics Types
export interface SystemStats {
  vector_database: {
    total_scenes: number;
    indices: {
      text: number;
      image: number;
      audio: number;
      fusion: number;
    };
  };
  metadata_database: {
    status: string;
    document_count?: number;
    index_size?: number;
  };
  models: {
    phobert_loaded: boolean;
    clip_loaded: boolean;
    asr_loaded: boolean;
    cross_encoder_loaded: boolean;
  };
  llm_service: {
    openai_available: boolean;
  };
  system_info: {
    version: string;
    vector_dimension: number;
  };
}

// Performance Monitoring
export interface PerformanceMetrics {
  searchTime: number;
  renderTime: number;
  apiLatency: number;
  cacheHitRate: number;
}
