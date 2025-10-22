import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bell, 
  Settings, 
  Sun, 
  Moon, 
  Globe,
  X,
  CheckCircle,
  AlertCircle,
  Info,
  XCircle
} from 'lucide-react';
import { Home } from '@/pages/Home';
import { SearchResults } from '@/pages/SearchResults';
import { FramesCliDemo } from '@/pages/FramesCliDemo';
import { useSearchStore, useUIStore, useSystemStore } from '@/store';
import { checkHealth } from '@/api/searchApi';

// Notification Component
const Notification: React.FC<{
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: Date;
  onRemove: (id: string) => void;
}> = ({ id, type, message, timestamp, onRemove }) => {
  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertCircle,
    info: Info
  };

  const colors = {
    success: 'bg-green-50 border-green-200 text-green-800',
    error: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800'
  };

  const Icon = icons[type];

  return (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className={`border rounded-lg p-4 shadow-lg ${colors[type]}`}
    >
      <div className="flex items-start space-x-3">
        <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-sm font-medium">{message}</p>
          <p className="text-xs opacity-75 mt-1">
            {timestamp.toLocaleTimeString('vi-VN')}
          </p>
        </div>
        <button
          onClick={() => onRemove(id)}
          className="opacity-60 hover:opacity-100 transition-opacity"
        >
          <X size={16} />
        </button>
      </div>
    </motion.div>
  );
};

// Notification Container
const NotificationContainer: React.FC = () => {
  const notifications = useUIStore(state => state.notifications);
  const removeNotification = useUIStore(state => state.removeNotification);

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      <AnimatePresence>
        {notifications.map((notification) => (
          <Notification
            key={notification.id}
            {...notification}
            onRemove={removeNotification}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

// Header Component
const Header: React.FC = () => {
  const theme = useUIStore(state => state.theme);
  const language = useUIStore(state => state.language);
  const sidebarOpen = useUIStore(state => state.sidebarOpen);
  
  const setTheme = useUIStore(state => state.setTheme);
  const setLanguage = useUIStore(state => state.setLanguage);
  const setSidebarOpen = useUIStore(state => state.setSidebarOpen);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const toggleLanguage = () => {
    setLanguage(language === 'vi' ? 'en' : 'vi');
  };

  return (
    <header className="bg-white dark:bg-gray-900 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-4">
            <div className="w-8 h-8 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">AI</span>
            </div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              Video Search
            </h1>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <a
              href="/"
              className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Trang chủ
            </a>
            <a
              href="/search"
              className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Tìm kiếm
            </a>
            <a
              href="/about"
              className="text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              Giới thiệu
            </a>
          </nav>

          {/* Actions */}
          <div className="flex items-center space-x-4">
            {/* Language Toggle */}
            <button
              onClick={toggleLanguage}
              className="p-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              title="Chuyển đổi ngôn ngữ"
            >
              <Globe size={20} />
            </button>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              title="Chuyển đổi giao diện"
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>

            {/* Settings */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              title="Cài đặt"
            >
              <Settings size={20} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

// Sidebar Component
const Sidebar: React.FC = () => {
  const sidebarOpen = useUIStore(state => state.sidebarOpen);
  const setSidebarOpen = useUIStore(state => state.setSidebarOpen);
  const clearHistory = useSearchStore(state => state.clearHistory);
  const searchHistory = useSearchStore(state => state.searchHistory);

  return (
    <AnimatePresence>
      {sidebarOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            onClick={() => setSidebarOpen(false)}
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            className="fixed top-0 right-0 h-full w-80 bg-white dark:bg-gray-900 shadow-xl z-50"
          >
            <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Cài đặt
              </h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-4 space-y-6">
              {/* Search History */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    Lịch sử tìm kiếm
                  </h3>
                  {searchHistory.length > 0 && (
                    <button
                      onClick={clearHistory}
                      className="text-sm text-red-600 hover:text-red-700"
                    >
                      Xóa tất cả
                    </button>
                  )}
                </div>
                
                <div className="space-y-2">
                  {searchHistory.length > 0 ? (
                    searchHistory.slice(0, 10).map((item) => (
                      <div
                        key={item.id}
                        className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                      >
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {item.query}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {new Date(item.timestamp).toLocaleDateString('vi-VN')} • {item.type}
                        </p>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Chưa có lịch sử tìm kiếm
                    </p>
                  )}
                </div>
              </div>

              {/* System Info */}
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-3">
                  Thông tin hệ thống
                </h3>
                <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <p>Phiên bản: 1.0.0</p>
                  <p>Ngôn ngữ: Tiếng Việt</p>
                  <p>Chế độ: Offline</p>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

// Main App Component
export const App: React.FC = () => {
  const theme = useUIStore(state => state.theme);
  //const setOnline = useSystemStore(state => state.setOnline);

  // Apply theme
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // Health check
  // useEffect(() => {
  //   const checkSystemHealth = async () => {
  //     try {
  //       await checkHealth();
  //       setOnline(true);
  //     } catch (error) {
  //       console.error('System health check failed:', error);
  //       setOnline(false);
  //     }
  //   };

  //   checkSystemHealth();
  //   const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds

  //   return () => clearInterval(interval);
  // }, [setOnline]);

  return (
    <Router>
      <div className={`min-h-screen ${theme === 'dark' ? 'dark' : ''}`}>
        <div className="bg-gray-50 dark:bg-gray-900 min-h-screen">
          <Header />
          
          <main>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/search" element={<SearchResults />} />
              <Route path="/frames-cli" element={<FramesCliDemo />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          <Sidebar />
          <NotificationContainer />
        </div>
      </div>
    </Router>
  );
};
