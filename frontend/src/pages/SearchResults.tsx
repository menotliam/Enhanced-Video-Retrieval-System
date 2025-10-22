import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Grid, 
  List, 
  Filter, 
  SortAsc, 
  SortDesc,
  Clock,
  Star,
  Eye,
  ArrowLeft,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Info,
  X,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { SearchBar } from '@/components/SearchBar';
import { ScenePreview } from '@/components/ScenePreview';
import { VideoPlayer } from '@/components/VideoPlayer';
import { Filters } from '@/components/Filters';
import { SearchQuery, SearchResult, Scene, FilterOptions } from '@/types';
import { useSearchStore, useUIStore } from '@/store';
import { performSearchWithCache, getFilterOptions } from '@/api/searchApi';

export const SearchResults: React.FC = () => {
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    objects: [],
    locations: [],
    time_of_day: [],
    weather: []
  });
  const [showFilters, setShowFilters] = useState(false);
  const [showVideoPlayer, setShowVideoPlayer] = useState(false);
  const [localSortBy, setLocalSortBy] = useState<'relevance' | 'time' | 'quality'>('relevance');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [localIsLoading, setLocalIsLoading] = useState(false);

  const searchResults = useSearchStore(state => state.searchResults);
  const currentQuery = useSearchStore(state => state.currentQuery);
  const searchType = useSearchStore(state => state.searchType);
  const viewMode = useSearchStore(state => state.viewMode);
  const activeFilters = useSearchStore(state => state.activeFilters);
  const selectedScene = useSearchStore(state => state.selectedScene);
  const isLoading: boolean = useSearchStore(state => state.isLoading);
  const error = useSearchStore(state => state.error);
  
  const setViewMode = useSearchStore(state => state.setViewMode);
  const setSortBy = useSearchStore(state => state.setSortBy);
  const setActiveFilters = useSearchStore(state => state.setActiveFilters);
  const setSelectedScene = useSearchStore(state => state.setSelectedScene);
  const setSearchResults = useSearchStore(state => state.setSearchResults);
  const setLoading = useSearchStore(state => state.setLoading);
  const setError = useSearchStore(state => state.setError);
  const addNotification = useUIStore(state => state.addNotification);

  // Load filter options
  useEffect(() => {
    const loadFilterOptions = async () => {
      try {
        const options = await getFilterOptions();
        setFilterOptions(options);
      } catch (error) {
        console.error('Failed to load filter options:', error);
      }
    };

    loadFilterOptions();
  }, []);

  // Handle search
  const handleSearch = async (searchQuery: SearchQuery) => {
    setLocalIsLoading(true);
    setLoading(true);
    setError(null);

    try {
      const result = await performSearchWithCache(searchQuery);
      setSearchResults(result);
      
      addNotification({
        type: 'success',
        message: `Tìm thấy ${result.total_results} kết quả trong ${result.search_time.toFixed(2)}s`
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Có lỗi xảy ra khi tìm kiếm';
      setError(errorMessage);
      addNotification({
        type: 'error',
        message: errorMessage
      });
    } finally {
      setLocalIsLoading(false);
      setLoading(false);
    }
  };

  // Handle scene click
  const handleSceneClick = (scene: Scene) => {
    setSelectedScene(scene);
    setShowVideoPlayer(true);
  };

  // Handle filters change
  const handleFiltersChange = (filters: any) => {
    setActiveFilters(filters);
    // Trigger new search with filters
    if (searchResults) {
      const newSearchQuery: SearchQuery = {
        query: currentQuery,
        type: searchType,
        filters: filters,
        top_k: 20
      };
      handleSearch(newSearchQuery);
    }
  };

  // Sort scenes
  const getSortedScenes = (scenes: Scene[]) => {
    return [...scenes].sort((a, b) => {
      let comparison = 0;
      
      switch (localSortBy) {
        case 'relevance':
          comparison = b.score - a.score;
          break;
        case 'time':
          comparison = a.start_time - b.start_time;
          break;
        case 'quality':
          const aQuality = (a.metadata.cross_encoder_score + a.metadata.fuzzy_score + a.metadata.quality_score) / 3;
          const bQuality = (b.metadata.cross_encoder_score + b.metadata.fuzzy_score + b.metadata.quality_score) / 3;
          comparison = bQuality - aQuality;
          break;
      }
      
      return sortOrder === 'asc' ? -comparison : comparison;
    });
  };

  // Get highlight terms from current query
  const getHighlightTerms = () => {
    if (!currentQuery) return [];
    return currentQuery.split(' ').filter(term => term.length > 2);
  };

  // Format search stats
  const formatSearchStats = () => {
    if (!searchResults) return null;
    
    return {
      totalResults: searchResults.total_results,
      searchTime: searchResults.search_time,
      query: searchResults.query
    };
  };

  const stats = formatSearchStats();
  const sortedScenes = searchResults ? getSortedScenes(searchResults.scenes) : [];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.history.back()}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft size={20} />
              </button>
              <h1 className="text-lg font-semibold text-gray-900">
                Kết quả tìm kiếm
              </h1>
            </div>

            {/* Search Bar */}
            <div className="flex-1 max-w-2xl mx-8">
              <SearchBar
                onSearch={handleSearch}
                isLoading={isLoading}
                placeholder="Tìm kiếm lại..."
              />
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setViewMode('grid' as any)}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'grid' 
                    ? 'bg-primary-100 text-primary-600' 
                    : 'text-gray-400 hover:text-gray-600'
                }`}
              >
                <Grid size={20} />
              </button>
              <button
                onClick={() => setViewMode('list' as any)}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'list' 
                    ? 'bg-primary-100 text-primary-600' 
                    : 'text-gray-400 hover:text-gray-600'
                }`}
              >
                <List size={20} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Stats and Controls */}
        {stats && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 mb-8"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="text-green-500" size={20} />
                  <span className="text-lg font-semibold text-gray-900">
                    {stats.totalResults.toLocaleString()} kết quả
                  </span>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                  <Clock size={14} />
                  <span>{stats.searchTime.toFixed(2)}s</span>
                </div>
                <div className="text-sm text-gray-500">
                  "{stats.query}"
                </div>
              </div>

              <div className="flex items-center space-x-4">
                {/* Sort Controls */}
                <div className="flex items-center space-x-2">
                  <select
                    value={localSortBy}
                    onChange={(e) => setLocalSortBy(e.target.value as any)}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                  >
                    <option value="relevance">Độ liên quan</option>
                    <option value="time">Thời gian</option>
                    <option value="quality">Chất lượng</option>
                  </select>
                  <button
                    onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                    className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    {sortOrder === 'asc' ? <SortAsc size={16} /> : <SortDesc size={16} />}
                  </button>
                </div>

                {/* Filters Toggle */}
                <Filters
                  filters={activeFilters}
                  options={filterOptions}
                  onFiltersChange={handleFiltersChange}
                  isOpen={showFilters}
                  onToggle={() => setShowFilters(!showFilters)}
                />
              </div>
            </div>

            {/* Active Filters Display */}
            {Object.keys(activeFilters).length > 0 && (
              <div className="flex items-center space-x-2 flex-wrap">
                <span className="text-sm text-gray-500">Bộ lọc:</span>
                {activeFilters.objects?.map((obj, idx) => (
                  <span
                    key={`obj-${idx}`}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-700"
                  >
                    {obj}
                    <button
                      onClick={() => {
                        const newFilters = { ...activeFilters };
                        newFilters.objects = newFilters.objects?.filter(item => item !== obj);
                        handleFiltersChange(newFilters);
                      }}
                      className="ml-1 hover:text-blue-900"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
                {activeFilters.locations?.map((loc, idx) => (
                  <span
                    key={`loc-${idx}`}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-700"
                  >
                    {loc}
                    <button
                      onClick={() => {
                        const newFilters = { ...activeFilters };
                        newFilters.locations = newFilters.locations?.filter(item => item !== loc);
                        handleFiltersChange(newFilters);
                      }}
                      className="ml-1 hover:text-green-900"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
                <button
                  onClick={() => handleFiltersChange({})}
                  className="text-xs text-red-600 hover:text-red-700"
                >
                  Xóa tất cả
                </button>
              </div>
            )}
          </motion.div>
        )}

        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8"
          >
            <div className="flex items-center space-x-3">
              <AlertCircle className="text-red-500" size={20} />
              <div>
                <h3 className="text-red-800 font-medium">Lỗi tìm kiếm</h3>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Loading State */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center justify-center py-12"
          >
            <div className="text-center">
              <RefreshCw className="animate-spin text-primary-600 mx-auto mb-4" size={32} />
              <p className="text-gray-600">Đang tìm kiếm...</p>
            </div>
          </motion.div>
        )}

        {/* Results Grid */}
        {!isLoading && searchResults && (
          <AnimatePresence>
            {sortedScenes.length > 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={viewMode === 'grid' 
                  ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
                  : 'space-y-4'
                }
              >
                {sortedScenes.map((scene, index) => (
                  <ScenePreview
                     key={scene.scene_id}
                     scene={scene}
                     onClick={handleSceneClick}
                     viewMode={viewMode === 'timeline' ? 'list' : viewMode}
                     highlightTerms={getHighlightTerms()}
                     index={index}
                   />
                ))}
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
                <Info className="text-gray-400 mx-auto mb-4" size={48} />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Không tìm thấy kết quả
                </h3>
                <p className="text-gray-600 mb-6">
                  Thử thay đổi từ khóa tìm kiếm hoặc bộ lọc để có kết quả tốt hơn
                </p>
                <button
                  onClick={() => handleFiltersChange({})}
                  className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
                >
                  Xóa bộ lọc
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        )}

        {/* RAG Answer */}
        {searchResults?.rag_answer && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 bg-gradient-to-r from-primary-50 to-accent-50 border border-primary-200 rounded-xl p-6"
          >
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <Star className="text-primary-600" size={16} />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">
                  Câu trả lời AI
                </h3>
                <p className="text-gray-700 leading-relaxed">
                  {searchResults.rag_answer}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </main>

      {/* Video Player Modal */}
      <AnimatePresence>
        {showVideoPlayer && selectedScene && (
          <VideoPlayer
            scene={selectedScene}
            autoPlay={true}
            onClose={() => {
              setShowVideoPlayer(false);
              setSelectedScene(null);
            }}
            showFullscreen={true}
          />
        )}
      </AnimatePresence>
    </div>
  );
};
