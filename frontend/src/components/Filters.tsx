import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Filter, 
  X, 
  Search, 
  Clock, 
  MapPin, 
  Tag, 
  Cloud, 
  Calendar,
  ChevronDown,
  ChevronUp,
  Sliders,
  RefreshCw,
  Check,
  Plus,
  Minus
} from 'lucide-react';
import { SearchFilters, FilterOptions } from '@/types';
import { useSearchStore } from '@/store';
import { format } from 'date-fns';
import { vi } from 'date-fns/locale';

interface FiltersProps {
  filters: SearchFilters;
  options: FilterOptions;
  onFiltersChange: (filters: SearchFilters) => void;
  isOpen: boolean;
  onToggle: () => void;
  className?: string;
}

export const Filters: React.FC<FiltersProps> = ({
  filters,
  options,
  onFiltersChange,
  isOpen,
  onToggle,
  className = ""
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['objects', 'locations']));
  const [searchTerms, setSearchTerms] = useState<Record<string, string>>({});
  const [selectedRanges, setSelectedRanges] = useState<Record<string, [number, number]>>({});

  const setActiveFilters = useSearchStore(state => state.setActiveFilters);

  // Update store when filters change
  useEffect(() => {
    setActiveFilters(filters);
  }, [filters, setActiveFilters]);

  // Toggle section expansion
  const toggleSection = useCallback((section: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  }, []);

  // Handle filter changes
  const updateFilter = useCallback((key: keyof SearchFilters, value: any) => {
    const newFilters = { ...filters, [key]: value };
    onFiltersChange(newFilters);
  }, [filters, onFiltersChange]);

  // Handle array filter changes
  const updateArrayFilter = useCallback((key: keyof SearchFilters, value: string, checked: boolean) => {
    const currentArray = (filters[key] as string[]) || [];
    const newArray = checked
      ? [...currentArray, value]
      : currentArray.filter(item => item !== value);
    
    updateFilter(key, newArray);
  }, [filters, updateFilter]);

  // Handle date range changes
  const updateDateRange = useCallback((start: Date | null, end: Date | null) => {
    updateFilter('date_range', start && end ? { start, end } : undefined);
  }, [updateFilter]);

  // Clear all filters
  const clearAllFilters = useCallback(() => {
    onFiltersChange({});
  }, [onFiltersChange]);

  // Get active filter count
  const getActiveFilterCount = useCallback(() => {
    let count = 0;
    if (filters.objects?.length) count += filters.objects.length;
    if (filters.locations?.length) count += filters.locations.length;
    if (filters.time_of_day?.length) count += filters.time_of_day.length;
    if (filters.weather?.length) count += filters.weather.length;
    if (filters.date_range) count += 1;
    return count;
  }, [filters]);

  // Filter options by search term
  const getFilteredOptions = useCallback((options: string[], section: string) => {
    const searchTerm = searchTerms[section]?.toLowerCase() || '';
    return options.filter(option => 
      option.toLowerCase().includes(searchTerm)
    );
  }, [searchTerms]);

  // Render filter section
  const renderFilterSection = (
    title: string,
    key: keyof SearchFilters,
    options: string[],
    icon: React.ReactNode,
    section: string
  ) => {
    const isExpanded = expandedSections.has(section);
    const selectedValues = (filters[key] as string[]) || [];
    const filteredOptions = getFilteredOptions(options, section);

    return (
      <div className="border-b border-gray-200 last:border-b-0">
        <button
          onClick={() => toggleSection(section)}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-center space-x-3">
            {icon}
            <span className="font-medium text-gray-900">{title}</span>
            {selectedValues.length > 0 && (
              <span className="bg-primary-100 text-primary-600 text-xs px-2 py-1 rounded-full">
                {selectedValues.length}
              </span>
            )}
          </div>
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="p-4 pt-0">
                {/* Search input */}
                <div className="relative mb-3">
                  <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder={`Tìm kiếm ${title.toLowerCase()}...`}
                    value={searchTerms[section] || ''}
                    onChange={(e) => setSearchTerms(prev => ({ ...prev, [section]: e.target.value }))}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                  />
                </div>

                {/* Options */}
                <div className="max-h-48 overflow-y-auto space-y-2">
                  {filteredOptions.length > 0 ? (
                    filteredOptions.map((option) => {
                      const isSelected = selectedValues.includes(option);
                      return (
                        <label
                          key={option}
                          className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors"
                        >
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={(e) => updateArrayFilter(key, option, e.target.checked)}
                            className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                          />
                          <span className="text-sm text-gray-700 flex-1">{option}</span>
                          {isSelected && <Check size={14} className="text-primary-600" />}
                        </label>
                      );
                    })
                  ) : (
                    <div className="text-sm text-gray-500 text-center py-4">
                      Không tìm thấy kết quả
                    </div>
                  )}
                </div>

                {/* Select all/none */}
                {filteredOptions.length > 0 && (
                  <div className="flex space-x-2 mt-3 pt-3 border-t border-gray-100">
                    <button
                      onClick={() => updateFilter(key, filteredOptions)}
                      className="text-xs text-primary-600 hover:text-primary-700"
                    >
                      Chọn tất cả
                    </button>
                    <button
                      onClick={() => updateFilter(key, [])}
                      className="text-xs text-gray-500 hover:text-gray-700"
                    >
                      Bỏ chọn tất cả
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <div className={`relative ${className}`}>
      {/* Filter Toggle Button */}
      <button
        onClick={onToggle}
        className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
      >
        <Sliders size={16} />
        <span className="font-medium">Bộ lọc</span>
        {getActiveFilterCount() > 0 && (
          <span className="bg-primary-600 text-white text-xs px-2 py-1 rounded-full">
            {getActiveFilterCount()}
          </span>
        )}
      </button>

      {/* Filters Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 mt-2 w-80 bg-white border border-gray-200 rounded-xl shadow-lg z-50"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div className="flex items-center space-x-2">
                <Filter size={16} />
                <span className="font-semibold text-gray-900">Bộ lọc tìm kiếm</span>
              </div>
              <div className="flex items-center space-x-2">
                {getActiveFilterCount() > 0 && (
                  <button
                    onClick={clearAllFilters}
                    className="text-xs text-gray-500 hover:text-gray-700"
                  >
                    Xóa tất cả
                  </button>
                )}
                <button
                  onClick={onToggle}
                  className="p-1 hover:bg-gray-100 rounded"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* Filter Sections */}
            <div className="max-h-96 overflow-y-auto">
              {/* Objects */}
              {renderFilterSection(
                'Đối tượng',
                'objects',
                options.objects,
                <Tag size={16} className="text-blue-500" />,
                'objects'
              )}

              {/* Locations */}
              {renderFilterSection(
                'Địa điểm',
                'locations',
                options.locations,
                <MapPin size={16} className="text-green-500" />,
                'locations'
              )}

              {/* Time of Day */}
              {renderFilterSection(
                'Thời gian trong ngày',
                'time_of_day',
                options.time_of_day,
                <Clock size={16} className="text-orange-500" />,
                'time_of_day'
              )}

              {/* Weather */}
              {renderFilterSection(
                'Thời tiết',
                'weather',
                options.weather,
                <Cloud size={16} className="text-gray-500" />,
                'weather'
              )}

              {/* Date Range */}
              <div className="border-b border-gray-200 last:border-b-0">
                <button
                  onClick={() => toggleSection('date_range')}
                  className="w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <Calendar size={16} className="text-purple-500" />
                    <span className="font-medium text-gray-900">Khoảng thời gian</span>
                    {filters.date_range && (
                      <span className="bg-primary-100 text-primary-600 text-xs px-2 py-1 rounded-full">
                        1
                      </span>
                    )}
                  </div>
                  {expandedSections.has('date_range') ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>

                <AnimatePresence>
                  {expandedSections.has('date_range') && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="p-4 pt-0">
                        <div className="space-y-3">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                              Từ ngày
                            </label>
                            <input
                              type="date"
                              value={filters.date_range?.start ? format(filters.date_range.start, 'yyyy-MM-dd') : ''}
                              onChange={(e) => {
                                const start = e.target.value ? new Date(e.target.value) : null;
                                const end = filters.date_range?.end || null;
                                updateDateRange(start, end);
                              }}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                              Đến ngày
                            </label>
                            <input
                              type="date"
                              value={filters.date_range?.end ? format(filters.date_range.end, 'yyyy-MM-dd') : ''}
                              onChange={(e) => {
                                const start = filters.date_range?.start || null;
                                const end = e.target.value ? new Date(e.target.value) : null;
                                updateDateRange(start, end);
                              }}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                            />
                          </div>
                          {filters.date_range && (
                            <button
                              onClick={() => updateDateRange(null, null)}
                              className="text-xs text-red-600 hover:text-red-700"
                            >
                              Xóa khoảng thời gian
                            </button>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-200 bg-gray-50">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>{getActiveFilterCount()} bộ lọc đang hoạt động</span>
                <button
                  onClick={clearAllFilters}
                  className="text-primary-600 hover:text-primary-700 font-medium"
                >
                  Xóa tất cả
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
