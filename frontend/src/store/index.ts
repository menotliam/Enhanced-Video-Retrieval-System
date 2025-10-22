import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';
import { 
  SearchResult, 
  Scene, 
  SearchQuery, 
  SearchFilters, 
  UIState, 
  SearchHistoryItem, 
  FilterOptions, 
  SystemStats,
  SearchType 
} from '@/types';

// Main Search Store
interface SearchStore {
  // Search state
  currentQuery: string;
  searchType: SearchType;
  searchResults: SearchResult | null;
  isLoading: boolean;
  error: string | null;
  
  // UI state
  selectedScene: Scene | null;
  viewMode: 'grid' | 'list' | 'timeline';
  sortBy: 'relevance' | 'time' | 'quality';
  filterPanelOpen: boolean;
  
  // Search history
  searchHistory: SearchHistoryItem[];
  
  // Filters
  activeFilters: SearchFilters;
  filterOptions: FilterOptions;
  
  // Performance
  lastSearchTime: number;
  
  // Actions
  setCurrentQuery: (query: string) => void;
  setSearchType: (type: SearchType) => void;
  setSearchResults: (results: SearchResult | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedScene: (scene: Scene | null) => void;
  setViewMode: (mode: 'grid' | 'list' | 'timeline') => void;
  setSortBy: (sort: 'relevance' | 'time' | 'quality') => void;
  setFilterPanelOpen: (open: boolean) => void;
  setActiveFilters: (filters: SearchFilters) => void;
  addToHistory: (item: Omit<SearchHistoryItem, 'id' | 'timestamp'>) => void;
  clearHistory: () => void;
  clearResults: () => void;
  reset: () => void;
}

export const useSearchStore = create<SearchStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentQuery: '',
        searchType: 'text',
        searchResults: null,
        isLoading: false,
        error: null,
        selectedScene: null,
        viewMode: 'grid',
        sortBy: 'relevance',
        filterPanelOpen: false,
        searchHistory: [],
        activeFilters: {},
        filterOptions: {
          objects: [],
          locations: [],
          time_of_day: [],
          weather: []
        },
        lastSearchTime: 0,

        // Actions
        setCurrentQuery: (query) => set({ currentQuery: query }),
        
        setSearchType: (type) => set({ 
          searchType: type,
          error: null 
        }),
        
        setSearchResults: (results) => set({ 
          searchResults: results,
          lastSearchTime: results?.search_time || 0,
          error: null 
        }),
        
        setLoading: (loading) => set({ isLoading: loading }),
        
        setError: (error) => set({ 
          error, 
          isLoading: false 
        }),
        
        setSelectedScene: (scene) => set({ selectedScene: scene }),
        
        setViewMode: (mode) => set({ viewMode: mode }),
        
        setSortBy: (sort) => set({ sortBy: sort }),
        
        setFilterPanelOpen: (open) => set({ filterPanelOpen: open }),
        
        setActiveFilters: (filters) => set({ activeFilters: filters }),
        
        addToHistory: (item) => {
          const history = get().searchHistory;
          const newItem: SearchHistoryItem = {
            ...item,
            id: Date.now().toString(),
            timestamp: new Date(),
          };
          
          // Keep only last 10 searches
          const updatedHistory = [newItem, ...history.slice(0, 9)];
          set({ searchHistory: updatedHistory });
        },
        
        clearHistory: () => set({ searchHistory: [] }),
        
        clearResults: () => set({ 
          searchResults: null, 
          selectedScene: null, 
          error: null 
        }),
        
        reset: () => set({
          currentQuery: '',
          searchType: 'text',
          searchResults: null,
          isLoading: false,
          error: null,
          selectedScene: null,
          activeFilters: {},
          filterPanelOpen: false,
        }),
      }),
      {
        name: 'ai-video-search-store',
        partialize: (state) => ({
          searchHistory: state.searchHistory,
          viewMode: state.viewMode,
          sortBy: state.sortBy,
          activeFilters: state.activeFilters,
        }),
      }
    ),
    { name: 'search-store' }
  )
);

// System Status Store
interface SystemStore {
  stats: SystemStats | null;
  isOnline: boolean;
  lastHealthCheck: Date | null;
  
  setStats: (stats: SystemStats) => void;
  setOnline: (online: boolean) => void;
  updateHealthCheck: () => void;
}

export const useSystemStore = create<SystemStore>()(
  devtools(
    (set) => ({
      stats: null,
      isOnline: true,
      lastHealthCheck: null,
      
      setStats: (stats) => set({ stats }),
      setOnline: (online) => set({ isOnline: online }),
      updateHealthCheck: () => set({ lastHealthCheck: new Date() }),
    }),
    { name: 'system-store' }
  )
);

// Audio Recording Store
interface AudioStore {
  isRecording: boolean;
  audioBlob: Blob | null;
  duration: number;
  recordingError: string | null;
  
  setRecording: (recording: boolean) => void;
  setAudioBlob: (blob: Blob | null) => void;
  setDuration: (duration: number) => void;
  setRecordingError: (error: string | null) => void;
  reset: () => void;
}

export const useAudioStore = create<AudioStore>()(
  devtools(
    (set) => ({
      isRecording: false,
      audioBlob: null,
      duration: 0,
      recordingError: null,
      
      setRecording: (recording) => set({ isRecording: recording }),
      setAudioBlob: (blob) => set({ audioBlob: blob }),
      setDuration: (duration) => set({ duration }),
      setRecordingError: (error) => set({ recordingError: error }),
      reset: () => set({
        isRecording: false,
        audioBlob: null,
        duration: 0,
        recordingError: null,
      }),
    }),
    { name: 'audio-store' }
  )
);

// UI Store for global UI state
interface UIStore {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  language: 'vi' | 'en';
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    timestamp: Date;
  }>;
  
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setLanguage: (language: 'vi' | 'en') => void;
  addNotification: (notification: Omit<UIStore['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useUIStore = create<UIStore>()(
  devtools(
    persist(
      (set, get) => ({
        sidebarOpen: false,
        theme: 'light',
        language: 'vi',
        notifications: [],
        
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        setTheme: (theme) => set({ theme }),
        setLanguage: (language) => set({ language }),
        
        addNotification: (notification) => {
          const notifications = get().notifications;
          const newNotification = {
            ...notification,
            id: Date.now().toString(),
            timestamp: new Date(),
          };
          
          set({ 
            notifications: [...notifications, newNotification].slice(-5) // Keep only 5 most recent
          });
        },
        
        removeNotification: (id) => {
          const notifications = get().notifications.filter(n => n.id !== id);
          set({ notifications });
        },
        
        clearNotifications: () => set({ notifications: [] }),
      }),
      {
        name: 'ui-store',
        partialize: (state) => ({
          theme: state.theme,
          language: state.language,
        }),
      }
    ),
    { name: 'ui-store' }
  )
);
