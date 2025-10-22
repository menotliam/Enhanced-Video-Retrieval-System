import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  Image, 
  Mic, 
  Sparkles, 
  TrendingUp, 
  Clock, 
  Star,
  Play,
  ArrowRight,
  Zap,
  Shield,
  Globe
} from 'lucide-react';
import { SearchBar } from '@/components/SearchBar';
import { SearchQuery, SearchResult, SystemStats } from '@/types';
import { useSearchStore, useSystemStore, useUIStore } from '@/store';
import { performSearchWithCache, getSystemStats } from '@/api/searchApi';

export const Home: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [recentSearches, setRecentSearches] = useState<Array<{query: string, type: string, timestamp: Date}>>([]);
  
  const searchHistory = useSearchStore(state => state.searchHistory);
  const setSearchResults = useSearchStore(state => state.setSearchResults);
  const setLoading = useSearchStore(state => state.setLoading);
  const setError = useSearchStore(state => state.setError);
  const addToHistory = useSearchStore(state => state.addToHistory);
  const language = useUIStore(state => state.language);
  const addNotification = useUIStore(state => state.addNotification);

  // Load system stats
  // useEffect(() => {
  //   const loadStats = async () => {
  //     try {
  //       const systemStats = await getSystemStats();
  //       setStats(systemStats);
  //     } catch (error) {
  //       console.error('Failed to load system stats:', error);
  //     }
  //   };

  //   loadStats();
  // }, []);

  // Handle search
  const handleSearch = async (searchQuery: SearchQuery) => {
    setIsLoading(true);
    setLoading(true);
    setError(null);

    try {
      const result = await performSearchWithCache(searchQuery);
      setSearchResults(result);
      
      // Add to history
      addToHistory({
        query: searchQuery.query,
        type: searchQuery.type,
        resultCount: result.total_results
      });

      // Show success notification
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
      setIsLoading(false);
      setLoading(false);
    }
  };

  // Example searches
  const exampleSearches = [
    { query: 'người đi bộ trên đường', type: 'text', icon: Search },
    { query: 'xe hơi đang chạy', type: 'text', icon: Search },
    { query: 'cảnh thành phố về đêm', type: 'text', icon: Search },
    { query: 'tiếng còi xe', type: 'audio', icon: Mic },
    { query: 'hình ảnh biển', type: 'image', icon: Image },
  ];

  // Features
  const features = [
    {
      icon: Search,
      title: 'Tìm kiếm đa phương tiện',
      description: 'Hỗ trợ tìm kiếm bằng văn bản, hình ảnh và giọng nói tiếng Việt'
    },
    {
      icon: Zap,
      title: 'Tốc độ cao',
      description: 'Tìm kiếm nhanh chóng với công nghệ AI tiên tiến'
    },
    {
      icon: Star,
      title: 'Độ chính xác cao',
      description: 'Kết quả tìm kiếm chính xác với thuật toán re-ranking thông minh'
    },
    {
      icon: Shield,
      title: 'Bảo mật tuyệt đối',
      description: 'Dữ liệu được xử lý hoàn toàn offline, đảm bảo quyền riêng tư'
    },
    {
      icon: Globe,
      title: 'Tối ưu cho tiếng Việt',
      description: 'Được thiết kế đặc biệt cho ngôn ngữ và văn hóa Việt Nam'
    },
    {
      icon: Play,
      title: 'Xem trước video',
      description: 'Xem trước các cảnh video với thời gian chính xác'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg flex items-center justify-center">
                  <Search className="text-white" size={20} />
                </div>
                <h1 className="text-xl font-bold text-gray-900">
                  AI Video Search
                </h1>
              </div>
              <div className="hidden md:flex items-center space-x-1 text-sm text-gray-500">
                <span>•</span>
                <span>Hệ thống tìm kiếm video thông minh</span>
              </div>
            </div>

            {/* System Status */}
            {stats && (
              <div className="hidden lg:flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${stats.llm_service.openai_available ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-gray-600">AI Service</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="text-gray-600">
                    {stats.vector_database.total_scenes.toLocaleString()} scenes
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-6xl font-bold text-gray-900 mb-6"
          >
            Tìm kiếm video
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-accent-600">
              {' '}thông minh
            </span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto"
          >
            Khám phá kho video khổng lồ với công nghệ AI tiên tiến. 
            Tìm kiếm bằng văn bản, hình ảnh hoặc giọng nói tiếng Việt.
          </motion.p>

          {/* Search Bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="max-w-4xl mx-auto"
          >
            <SearchBar
              onSearch={handleSearch}
              isLoading={isLoading}
              placeholder="Tìm kiếm video bằng văn bản, hình ảnh hoặc giọng nói..."
            />
          </motion.div>
        </div>

        {/* Quick Stats */}
        {stats && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
          >
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Play className="text-blue-600" size={20} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.vector_database.total_scenes.toLocaleString()}
                  </p>
                  <p className="text-sm text-gray-600">Scenes</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Zap className="text-green-600" size={20} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.vector_database.indices.fusion.toLocaleString()}
                  </p>
                  <p className="text-sm text-gray-600">Fusion Index</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Star className="text-purple-600" size={20} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.system_info.vector_dimension}
                  </p>
                  <p className="text-sm text-gray-600">Dimensions</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="text-orange-600" size={20} />
                </div>
                <div>
                  <p className="text-2xl font-bold text-gray-900">
                    {stats.models.phobert_loaded ? '4' : '3'}
                  </p>
                  <p className="text-sm text-gray-600">Models</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Example Searches */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            Thử ngay với các ví dụ
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {exampleSearches.map((example, index) => {
              const Icon = example.icon;
              return (
                <button
                  key={index}
                  onClick={() => handleSearch({
                    query: example.query,
                    type: example.type as any,
                    top_k: 20
                  })}
                  className="bg-white rounded-xl p-4 shadow-sm border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all text-left group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center group-hover:bg-primary-100 transition-colors">
                      <Icon className="text-gray-600 group-hover:text-primary-600" size={16} />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 group-hover:text-primary-600 transition-colors">
                        {example.query}
                      </p>
                      <p className="text-sm text-gray-500">
                        {example.type === 'text' ? 'Tìm kiếm văn bản' : 
                         example.type === 'image' ? 'Tìm kiếm hình ảnh' : 'Tìm kiếm âm thanh'}
                      </p>
                    </div>
                    <ArrowRight className="text-gray-400 group-hover:text-primary-600 transition-colors" size={16} />
                  </div>
                </button>
              );
            })}
          </div>
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">
            Tính năng nổi bật
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-all"
                >
                  <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="text-primary-600" size={24} />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Recent Searches */}
        {searchHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mb-16"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
              Tìm kiếm gần đây
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {searchHistory.slice(0, 6).map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleSearch({
                    query: item.query,
                    type: item.type,
                    top_k: 20
                  })}
                  className="bg-white rounded-xl p-4 shadow-sm border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all text-left group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Clock className="text-gray-400" size={14} />
                      <span className="text-sm text-gray-500">
                        {new Date(item.timestamp).toLocaleDateString('vi-VN')}
                      </span>
                    </div>
                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                      {item.type}
                    </span>
                  </div>
                  <p className="font-medium text-gray-900 group-hover:text-primary-600 transition-colors">
                    {item.query}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    {item.resultCount} kết quả
                  </p>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="text-center"
        >
          <div className="bg-gradient-to-r from-primary-600 to-accent-600 rounded-2xl p-8 text-white">
            <h2 className="text-2xl font-bold mb-4">
              Bắt đầu tìm kiếm ngay hôm nay
            </h2>
            <p className="text-lg mb-6 opacity-90">
              Khám phá kho video khổng lồ với công nghệ AI tiên tiến
            </p>
            <button
              onClick={() => document.getElementById('search-bar')?.scrollIntoView({ behavior: 'smooth' })}
              className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Tìm kiếm ngay
            </button>
          </div>
        </motion.div>
      </main>
    </div>
  );
};
