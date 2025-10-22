import React, { useState } from 'react';
import { Download, FileSpreadsheet, CheckCircle } from 'lucide-react';
import { searchFramesCLI, FramesCliResponse, searchAndExportCSV, CSVExportOptions } from '@/api/searchApi';

export const FramesCliDemo: React.FC = () => {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [data, setData] = useState<FramesCliResponse | null>(null);

  const onSearch = async () => {
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    setData(null);
    try {
      const resp = await searchFramesCLI(query.trim(), topK);
      setData(resp);
    } catch (e: any) {
      setError(e?.message || 'Search failed');
    } finally {
      setIsLoading(false);
    }
  };

  const onExportCSV = async () => {
    if (!query.trim()) {
      setError('Vui lòng nhập từ khóa tìm kiếm trước khi xuất CSV');
      return;
    }

    setIsExporting(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      const exportOptions: CSVExportOptions = {
        query: query.trim(),
        top_k: topK,
        filename: `search_${query.replace(/\s+/g, '_')}_${Date.now()}.csv`
      };
      
      await searchAndExportCSV(exportOptions);
      
      setSuccessMessage(`Đã xuất CSV thành công! File đã được tải về thư mục Downloads.`);
      
      // Auto-hide success message after 5 seconds
      setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
    } catch (e: any) {
      setError(e?.message || 'Export CSV failed');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">Frames CLI Search</h1>

        <div className="bg-white border border-gray-200 rounded-xl p-4 mb-6">
          <div className="flex flex-col md:flex-row md:items-center gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Nhập từ khóa tìm kiếm..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Top-k</label>
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Math.max(1, Math.min(100, Number(e.target.value))))}
                className="w-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={onSearch}
                disabled={isLoading || !query.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Đang tìm...
                  </>
                ) : (
                  'Tìm kiếm'
                )}
              </button>
              <button
                onClick={onExportCSV}
                disabled={isExporting || !query.trim()}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
                title="Xuất kết quả ra file CSV"
              >
                {isExporting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Đang xuất...
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    Xuất CSV
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4 mb-6">
            {error}
          </div>
        )}

        {successMessage && (
          <div className="bg-green-50 border border-green-200 text-green-800 rounded-lg p-4 mb-6 flex items-center gap-3">
            <CheckCircle size={20} className="flex-shrink-0" />
            <span>{successMessage}</span>
          </div>
        )}

        {data && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm text-gray-600">
                Tìm thấy {data.total_results} kết quả trong {data.search_time.toFixed(2)}s
              </p>
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <FileSpreadsheet size={16} />
                <span>CSV: video_id, frame_idx</span>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {data.results.map((item, idx) => (
                <div key={`${item.video_id}-${item.frame_idx}-${idx}`} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                  {item.image_url ? (
                    <img src={item.image_url} alt={`frame ${item.frame_idx}`} className="w-full h-40 object-cover" />
                  ) : (
                    <div className="w-full h-40 bg-gray-100 flex items-center justify-center text-gray-400 text-sm">No image</div>
                  )}
                  <div className="p-3 text-sm text-gray-800">
                    <div className="flex justify-between">
                      <span className="font-medium">Frame</span>
                      <span>#{item.frame_idx}</span>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="font-medium">Score</span>
                      <span>{item.score?.toFixed(4)}</span>
                    </div>
                    <div className="truncate text-gray-500 mt-1">{item.video_id}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


