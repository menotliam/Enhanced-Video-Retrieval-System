import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Mic, 
  MicOff, 
  Image, 
  Upload, 
  X, 
  History, 
  Sparkles,
  Loader2,
  Volume2,
  VolumeX
} from 'lucide-react';
import { SearchQuery, SearchType } from '@/types';
import { useSearchStore } from '@/store';
import { getSearchSuggestions } from '@/api/searchApi';

interface SearchBarProps {
  onSearch: (query: SearchQuery) => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  isLoading = false,
  placeholder = "Tìm kiếm video bằng văn bản, hình ảnh hoặc giọng nói...",
  className = ""
}) => {
  const [query, setQuery] = useState('');
  const [searchType, setLocalSearchType] = useState<SearchType>('text');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [useRAG, setUseRAG] = useState(false);
  
  const searchHistory = useSearchStore(state => state.searchHistory);
  const setCurrentQuery = useSearchStore(state => state.setCurrentQuery);
  const setGlobalSearchType = useSearchStore(state => state.setSearchType);
  const addToHistory = useSearchStore(state => state.addToHistory);

  const inputRef = useRef<HTMLInputElement>(null);
  const recordingIntervalRef = useRef<ReturnType<typeof setInterval>>();

  // Debounced search suggestions
  useEffect(() => {
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }

    const timeoutId = setTimeout(async () => {
      try {
        const result = await getSearchSuggestions(query, 5);
        setSuggestions(result.suggestions);
      } catch (error) {
        console.error('Failed to get suggestions:', error);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [query]);

  // Audio recording setup
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      
      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setRecordingTime(0);

      // Update recording time
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    }
  }, [mediaRecorder, isRecording]);

  // Image dropzone
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setLocalSearchType('image');
      setGlobalSearchType('image');
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [setLocalSearchType, setGlobalSearchType]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false
  });

  // Handle search submission
  const handleSearch = useCallback(() => {
    if (isLoading) return;

    let searchQuery: SearchQuery = {
      query: query.trim(),
      type: searchType,
      top_k: 20,
      use_rag: useRAG
    };

    if (searchType === 'image' && selectedImage) {
      searchQuery.image = selectedImage;
    } else if (searchType === 'audio' && audioBlob) {
      searchQuery.audio = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
    }

    if (searchQuery.query || searchQuery.image || searchQuery.audio) {
      setCurrentQuery(searchQuery.query);
      addToHistory({
        query: searchQuery.query,
        type: searchType,
        resultCount: 0
      });
      onSearch(searchQuery);
    }
  }, [query, searchType, selectedImage, audioBlob, useRAG, isLoading, setCurrentQuery, addToHistory, onSearch]);

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  // Handle history click
  const handleHistoryClick = (historyItem: any) => {
    setQuery(historyItem.query);
    setLocalSearchType(historyItem.type);
    setGlobalSearchType(historyItem.type);
    setShowSuggestions(false);
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  // Clear current input
  const clearInput = () => {
    setQuery('');
    setSelectedImage(null);
    setImagePreview('');
    setAudioBlob(null);
    setLocalSearchType('text');
    setGlobalSearchType('text');
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  // Format recording time
  const formatRecordingTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`relative w-full max-w-4xl mx-auto ${className}`}>
      {/* Search Type Tabs */}
      <div className="flex mb-4 space-x-1 bg-gray-100 rounded-lg p-1">
        {[
          { type: 'text', icon: Search, label: 'Văn bản' },
          { type: 'image', icon: Image, label: 'Hình ảnh' },
          { type: 'audio', icon: Mic, label: 'Giọng nói' }
        ].map(({ type, icon: Icon, label }) => (
          <button
            key={type}
            onClick={() => {
              setLocalSearchType(type as SearchType);
              setGlobalSearchType(type as SearchType);
              clearInput();
            }}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
              searchType === type
                ? 'bg-white shadow-sm text-primary-600'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Icon size={16} />
            <span className="text-sm font-medium">{label}</span>
          </button>
        ))}
      </div>

      {/* Main Search Input */}
      <div className="relative">
        {/* Text Search */}
        {searchType === 'text' && (
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setShowSuggestions(true)}
              onKeyPress={handleKeyPress}
              placeholder={placeholder}
              className="w-full px-4 py-3 pl-12 pr-20 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              disabled={isLoading}
            />
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            
            {/* RAG Toggle */}
            <button
              onClick={() => setUseRAG(!useRAG)}
              className={`absolute right-16 top-1/2 transform -translate-y-1/2 p-1 rounded-md transition-all ${
                useRAG ? 'bg-primary-100 text-primary-600' : 'text-gray-400 hover:text-gray-600'
              }`}
              title="Sử dụng AI để tạo câu trả lời"
            >
              <Sparkles size={16} />
            </button>

            {/* Search Button */}
            <button
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-primary-600 text-white p-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Search size={16} />
              )}
            </button>
          </div>
        )}

        {/* Image Search */}
        {searchType === 'image' && (
          <div className="space-y-4">
            {!selectedImage ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                  isDragActive
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-300 hover:border-primary-400'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto mb-4 text-gray-400" size={32} />
                <p className="text-lg font-medium text-gray-700 mb-2">
                  {isDragActive ? 'Thả ảnh vào đây' : 'Kéo thả ảnh hoặc click để chọn'}
                </p>
                <p className="text-sm text-gray-500">
                  Hỗ trợ: JPG, PNG, GIF, WebP
                </p>
              </div>
            ) : (
              <div className="relative">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-full h-48 object-cover rounded-xl"
                />
                <button
                  onClick={clearInput}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600 transition-all"
                >
                  <X size={16} />
                </button>
                <div className="mt-4 flex items-center space-x-4">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Mô tả thêm về ảnh (tùy chọn)..."
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isLoading}
                    className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-all"
                  >
                    {isLoading ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      'Tìm kiếm'
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Audio Search */}
        {searchType === 'audio' && (
          <div className="space-y-4">
            <div className="border border-gray-300 rounded-xl p-6">
              {!isRecording && !audioBlob ? (
                <div className="text-center">
                  <button
                    onClick={startRecording}
                    className="bg-red-500 text-white p-4 rounded-full hover:bg-red-600 transition-all mb-4"
                  >
                    <Mic size={24} />
                  </button>
                  <p className="text-lg font-medium text-gray-700">
                    Nhấn để bắt đầu ghi âm
                  </p>
                  <p className="text-sm text-gray-500">
                    Nói rõ ràng để có kết quả tốt nhất
                  </p>
                </div>
              ) : isRecording ? (
                <div className="text-center">
                  <button
                    onClick={stopRecording}
                    className="bg-gray-500 text-white p-4 rounded-full hover:bg-gray-600 transition-all mb-4"
                  >
                    <MicOff size={24} />
                  </button>
                  <p className="text-lg font-medium text-red-600 mb-2">
                    Đang ghi âm... {formatRecordingTime(recordingTime)}
                  </p>
                  <div className="w-16 h-1 bg-red-500 mx-auto rounded-full animate-pulse"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-center space-x-4">
                    <Volume2 size={20} className="text-gray-400" />
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div className="bg-primary-500 h-2 rounded-full w-full"></div>
                    </div>
                    <span className="text-sm text-gray-500">
                      {formatRecordingTime(recordingTime)}
                    </span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Mô tả thêm về âm thanh (tùy chọn)..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                    <button
                      onClick={handleSearch}
                      disabled={isLoading}
                      className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-all"
                    >
                      {isLoading ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        'Tìm kiếm'
                      )}
                    </button>
                    <button
                      onClick={clearInput}
                      className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-all"
                    >
                      Ghi lại
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Suggestions Dropdown */}
        <AnimatePresence>
          {showSuggestions && (suggestions.length > 0 || searchHistory.length > 0) && searchType === 'text' && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full left-0 right-0 mt-2 bg-white border border-gray-200 rounded-xl shadow-lg z-50 max-h-96 overflow-y-auto"
            >
              {/* Search Suggestions */}
              {suggestions.length > 0 && (
                <div className="p-2">
                  <div className="text-xs font-medium text-gray-500 px-3 py-1">Gợi ý tìm kiếm</div>
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="w-full text-left px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <Search size={14} className="inline mr-2 text-gray-400" />
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}

              {/* Search History */}
              {searchHistory.length > 0 && (
                <div className="p-2 border-t border-gray-100">
                  <div className="text-xs font-medium text-gray-500 px-3 py-1">Lịch sử tìm kiếm</div>
                  {searchHistory.slice(0, 5).map((item) => (
                    <button
                      key={item.id}
                      onClick={() => handleHistoryClick(item)}
                      className="w-full text-left px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <History size={14} className="inline mr-2 text-gray-400" />
                      {item.query}
                      <span className="text-xs text-gray-400 ml-2">
                        {new Date(item.timestamp).toLocaleDateString('vi-VN')}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
