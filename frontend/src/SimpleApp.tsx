import React, { useState } from 'react';
import { generateRAGAnswer, type FramesCliResponse, type RAGResponse } from './api/searchApi';

const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

type TabType = 'retrieval' | 'rag';

export const SimpleApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('retrieval');
  
  // Retrieval tab state
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [preFilter, setPreFilter] = useState(false);
  const [preFilterLimit, setPreFilterLimit] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [data, setData] = useState<FramesCliResponse | null>(null);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [csvFilename, setCsvFilename] = useState('');
  const [videoPreview, setVideoPreview] = useState<any | null>(null);
  
  // RAG tab state
  const [frameInput, setFrameInput] = useState('');
  const [ragQuery, setRagQuery] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);
  const [ragData, setRagData] = useState<RAGResponse | null>(null);

  const doSearch = async () => {
    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    setData(null);
    try {
      const params = new URLSearchParams({ query, top_k: String(topK) });
      if (preFilter) {
        params.set('pre_filter', 'true');
        if (preFilterLimit && preFilterLimit > 0) params.set('pre_filter_limit', String(preFilterLimit));
      }
      const resp = await fetch(`${API_BASE_URL}/search/frames-cli?${params.toString()}`);
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const json = (await resp.json()) as FramesCliResponse;
      setData(json);
    } catch (e: any) {
      setError(e?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const doExportCSV = async () => {
    if (!data || !data.results || data.results.length === 0) {
      setError('Không có kết quả để xuất CSV. Vui lòng tìm kiếm trước.');
      return;
    }

    setIsExporting(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      // Create CSV content from existing search results
      const csvContent = createCSVFromResults(data.results);
      const filename = sanitizeCsvFilename(csvFilename);
      
      // Download the CSV file
      downloadCSVFile(csvContent, filename);
      
      setSuccessMessage(`✅ Đã xuất CSV thành công! File: ${filename}`);
      
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

  // Helper function to create CSV content from search results
  const createCSVFromResults = (results: any[]) => {
    const csvRows: string[] = [];
    results.forEach(item => {
      csvRows.push(`${item.video_id || ''},${item.frame_idx || ''}`);
    });
    return csvRows.join('\n');
  };

  // Helper function to download CSV file
  const downloadCSVFile = (csvContent: string, filename: string) => {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Helper: sanitize and ensure .csv extension
  const sanitizeCsvFilename = (name: string): string => {
    let n = (name || '').trim();
    if (!n) return `search_${Date.now()}.csv`;
    // Replace characters not allowed in filenames on most OS
    n = n.replace(/[\\/:*?"<>|]/g, '_');
    if (!n.toLowerCase().endsWith('.csv')) n += '.csv';
    return n;
  };

  const doRAGSearch = async () => {
    setRagLoading(true);
    setRagError(null);
    setRagData(null);
    try {
      const response = await generateRAGAnswer(frameInput, ragQuery);
      setRagData(response);
    } catch (e: any) {
      setRagError(e?.response?.data?.detail || e?.message || 'RAG service error !');
    } finally {
      setRagLoading(false);
    }
  };

  const staticBase = API_BASE_URL.replace(/\/api\/v1$/, '');

  const buildYouTubeEmbedUrl = (watchUrl?: string | null, startSec?: number | null) => {
    if (!watchUrl) return null;
    try {
      const url = new URL(watchUrl);
      let videoId = '';
      if (url.hostname.includes('youtu.be')) {
        videoId = url.pathname.replace('/', '');
      } else if (url.searchParams.get('v')) {
        videoId = url.searchParams.get('v') || '';
      } else {
        // Fallback attempt to parse last path segment
        const parts = url.pathname.split('/');
        videoId = parts[parts.length - 1] || '';
      }
      const start = startSec != null ? Math.max(0, Math.floor(startSec)) : undefined;
      const params = new URLSearchParams();
      if (start !== undefined) params.set('start', String(start));
      params.set('autoplay', '1');
      params.set('rel', '0');
      const qp = params.toString();
      return `https://www.youtube.com/embed/${videoId}${qp ? `?${qp}` : ''}`;
    } catch {
      return null;
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '16px' }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 12 }}>AI Video Search System</h1>
        
        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid #e5e7eb' }}>
          <button
            onClick={() => setActiveTab('retrieval')}
            style={{
              padding: '8px 16px',
              background: activeTab === 'retrieval' ? '#2563eb' : 'transparent',
              color: activeTab === 'retrieval' ? '#fff' : '#6b7280',
              border: 'none',
              borderRadius: '8px 8px 0 0',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            Section 1: Retrieval
          </button>
          <button
            onClick={() => setActiveTab('rag')}
            style={{
              padding: '8px 16px',
              background: activeTab === 'rag' ? '#2563eb' : 'transparent',
              color: activeTab === 'rag' ? '#fff' : '#6b7280',
              border: 'none',
              borderRadius: '8px 8px 0 0',
              cursor: 'pointer',
              fontWeight: 500
            }}
          >
            Section 2: RAG
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'retrieval' && (
          <>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 10, marginBottom: 12 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'nowrap' }}>
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter query..."
                  style={{ flex: 1, minWidth: 200, padding: '8px 10px', border: '1px solid #d1d5db', borderRadius: 8 }}
                />
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: 14, color: '#6b7280' }}>Top-k</label>
                  <input
                    type="number"
                    min={1}
                    value={topK}
                    onChange={(e) => setTopK(Math.max(1, Number(e.target.value)))}
                    style={{ width: 80, padding: '10px 12px', border: '1px solid #d1d5db', borderRadius: 8 }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: 14, color: '#6b7280' }}>Pre-filter</label>
                  <label style={{ position: 'relative', display: 'inline-block', width: 44, height: 24 }}>
                    <input type="checkbox" checked={preFilter} onChange={(e) => setPreFilter(e.target.checked)} style={{ opacity: 0, width: 0, height: 0 }} />
                    <span style={{ position: 'absolute', cursor: 'pointer', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: preFilter ? '#2563eb' : '#d1d5db', transition: '.2s', borderRadius: 24 }} />
                    <span style={{ position: 'absolute', content: "''", height: 18, width: 18, left: preFilter ? 22 : 4, bottom: 3, backgroundColor: 'white', transition: '.2s', borderRadius: '50%' }} />
                  </label>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <label style={{ fontSize: 14, color: '#6b7280' }}>Limit</label>
                  <input
                    type="number"
                    min={0}
                    value={preFilterLimit}
                    onChange={(e) => setPreFilterLimit(Math.max(0, Number(e.target.value)))}
                    disabled={!preFilter}
                    style={{ width: 90, padding: '10px 12px', border: '1px solid #d1d5db', borderRadius: 8, opacity: preFilter ? 1 : 0.5 }}
                  />
                </div>
                <input
                  value={csvFilename}
                  onChange={(e) => setCsvFilename(e.target.value)}
                  placeholder="Filename (optional)"
                  title="Tên file CSV (tùy chọn)"
                  style={{ width: 200, padding: '10px 12px', border: '1px solid #d1d5db', borderRadius: 8 }}
                />
                <button
                  onClick={doSearch}
                  disabled={loading || !query.trim()}
                  style={{ padding: '10px 16px', background: '#2563eb', color: '#fff', border: 0, borderRadius: 8, cursor: 'pointer', opacity: loading || !query.trim() ? 0.6 : 1 }}
                >
                  {loading ? 'Searching...' : 'Search'}
                </button>
                <button
                  onClick={doExportCSV}
                  disabled={isExporting || !data || !data.results || data.results.length === 0}
                  style={{ padding: '10px 16px', background: '#16a34a', color: '#fff', border: 0, borderRadius: 8, cursor: 'pointer', opacity: isExporting || !data || !data.results || data.results.length === 0 ? 0.6 : 1 }}
                  title="Xuất kết quả đã tìm kiếm ra file CSV"
                >
                  {isExporting ? 'Exporting...' : '📄 Export CSV'}
                </button>
              </div>
            </div>
          </>
        )}

        {activeTab === 'rag' && (
          <>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10, padding: 10, marginBottom: 12 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'nowrap' }}>
                <input
                  value={frameInput}
                  onChange={(e) => setFrameInput(e.target.value)}
                  placeholder="Frame input (e.g., L21_V001:90)"
                  style={{ flex: 1, minWidth: 200, padding: '8px 10px', border: '1px solid #d1d5db', borderRadius: 8 }}
                />
                <input
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  placeholder="Enter question..."
                  style={{ flex: 1, minWidth: 200, padding: '8px 10px', border: '1px solid #d1d5db', borderRadius: 8 }}
                />
                <button
                  onClick={doRAGSearch}
                  disabled={ragLoading || !frameInput.trim() || !ragQuery.trim()}
                  style={{ padding: '10px 16px', background: '#2563eb', color: '#fff', border: 0, borderRadius: 8, cursor: 'pointer', opacity: ragLoading || !frameInput.trim() || !ragQuery.trim() ? 0.6 : 1 }}
                >
                  {ragLoading ? 'Generating...' : 'Generate Answer'}
                </button>
              </div>
            </div>
          </>
        )}

        {/* Error Messages */}
        {activeTab === 'retrieval' && error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: 12, padding: 12, marginBottom: 16 }}>{error}</div>
        )}
        
        {/* Success Messages */}
        {activeTab === 'retrieval' && successMessage && (
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', color: '#166534', borderRadius: 12, padding: 12, marginBottom: 16 }}>{successMessage}</div>
        )}
        
        {activeTab === 'rag' && ragError && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#991b1b', borderRadius: 12, padding: 12, marginBottom: 16 }}>{ragError}</div>
        )}

        {/* Retrieval Results */}
        {activeTab === 'retrieval' && data && (
          <div>
            {/* Inline YouTube preview panel removed; video preview opens in modal now */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <p style={{ color: '#6b7280' }}>
                Found {data.total_results} results in {data.search_time.toFixed(2)}s
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#6b7280' }}>
                <span>📄</span>
                <span>CSV: video_id, frame_idx</span>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
              {data.results.map((item, idx) => {
                const imgSrc = item.image_url
                  ? (item.image_url.startsWith('http') ? item.image_url : `${staticBase}${item.image_url}`)
                  : null;
                return (
                <div key={`${item.video_id}-${item.frame_idx}-${idx}`} style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, overflow: 'hidden' }}>
                  {imgSrc ? (
                    <img 
                      src={imgSrc} 
                      alt={`frame ${item.frame_idx}`} 
                      style={{ width: '100%', height: 140, objectFit: 'cover', display: 'block', cursor: 'zoom-in' }} 
                      onClick={() => setPreviewIndex(idx)}
                    />
                  ) : (
                    <div 
                      style={{ width: '100%', height: 140, background: '#f3f4f6', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af', fontSize: 12, cursor: 'default' }}
                      onClick={() => setPreviewIndex(idx)}
                    >
                      No image
                    </div>
                  )}
                  <div style={{ padding: 10, fontSize: 14, color: '#111827' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Frame</span>
                      <span>#{item.frame_idx}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                      <span>Score</span>
                      <span>{(item.score ?? 0).toFixed(4)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                      <span>Image</span>
                      <span>{item.image || '-'}</span>
                    </div>
                    <div style={{ color: '#6b7280', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.video_id}</div>
                    {item.frame_timestamp !== null && item.frame_timestamp !== undefined && (
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                        <span style={{ fontSize: 12, color: '#6b7280' }}>Time</span>
                        <span style={{ fontSize: 12, color: '#6b7280' }}>{Math.floor(item.frame_timestamp / 60)}:{(item.frame_timestamp % 60).toFixed(1).padStart(4, '0')}</span>
                      </div>
                    )}
                    {item.youtube_url && (
                      <div style={{ marginTop: 8, textAlign: 'center' }}>
                        <button
                          onClick={() => { setPreviewIndex(null); setVideoPreview(item); }}
                          style={{ 
                            display: 'inline-block',
                            padding: '6px 12px', 
                            background: '#ff0000', 
                            color: '#fff', 
                            border: 0,
                            borderRadius: 6, 
                            fontSize: 12,
                            fontWeight: 500,
                            cursor: 'pointer'
                          }}
                          title={item.frame_timestamp !== null && item.frame_timestamp !== undefined ? 'Preview Frame' : 'Preview Video'}
                        >
                          📺 {item.frame_timestamp !== null && item.frame_timestamp !== undefined ? 'Preview Frame' : 'Preview Video'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );})}
            </div>
          </div>
        )}

        {/* RAG Results */}
        {activeTab === 'rag' && ragData && (
          <div>
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 16, marginBottom: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#111827' }}>Generated Answer</h3>
              <div style={{ 
                background: '#f8fafc', 
                border: '1px solid #e5e7eb', 
                borderRadius: 8, 
                padding: 12, 
                fontSize: 14, 
                lineHeight: 1.6,
                color: '#374151',
                whiteSpace: 'pre-wrap'
              }}>
                {ragData.answer}
              </div>
              <div style={{ marginTop: 12, fontSize: 12, color: '#6b7280' }}>
                Processing time: {ragData.processing_time.toFixed(2)}s
              </div>
            </div>
            
            <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#111827' }}>Frame Information</h3>
              <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ color: '#6b7280' }}>Video ID:</span>
                    <span style={{ fontWeight: 500 }}>{ragData.frame_info.video_id}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ color: '#6b7280' }}>Frame Index:</span>
                    <span style={{ fontWeight: 500 }}>#{ragData.frame_info.frame_idx}</span>
                  </div>
                </div>
                <div style={{ flex: 1 }}>
                  {ragData.frame_info.image_url && (
                    <img 
                      src={ragData.frame_info.image_url.startsWith('http') ? ragData.frame_info.image_url : `${staticBase}${ragData.frame_info.image_url}`}
                      alt={`Frame ${ragData.frame_info.frame_idx}`}
                      style={{ 
                        width: '100%', 
                        maxWidth: 300, 
                        height: 200, 
                        objectFit: 'cover', 
                        borderRadius: 8,
                        border: '1px solid #e5e7eb'
                      }}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Inline Preview Block was moved into Retrieval; remove old preview tab block if any */}
        {false && (
          <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 12, padding: 16 }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: '#111827' }}>Video Preview</h3>
            {videoPreview && (
              <div>
                <div style={{ marginBottom: 8, color: '#6b7280', fontSize: 14 }}>
                  <span style={{ marginRight: 12 }}>Video: {videoPreview.video_id}</span>
                  <span>Frame: #{videoPreview.frame_idx}</span>
                </div>
                <div style={{ position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', borderRadius: 8, border: '1px solid #e5e7eb' }}>
                  {buildYouTubeEmbedUrl(videoPreview.youtube_url, videoPreview.frame_timestamp) ? (
                    <iframe
                      src={buildYouTubeEmbedUrl(videoPreview.youtube_url, videoPreview.frame_timestamp) || ''}
                      title="YouTube video player"
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0 }}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                      allowFullScreen
                    />
                  ) : (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>
                      No YouTube URL available
                    </div>
                  )}
                </div>
              </div>
            )}
            {!videoPreview && (
              <div style={{ color: '#6b7280' }}>Select a frame to preview its video.</div>
            )}
          </div>
        )}

        {/* Image Preview Modal - disabled when inline preview is used */}
        {previewIndex !== null && data && data.results[previewIndex] && !videoPreview && (() => {
          const preview = data.results[previewIndex];
          const imgSrc = preview.image_url ? (preview.image_url.startsWith('http') ? preview.image_url : `${staticBase}${preview.image_url}`) : null;
          const total = data.results.length;
          const goPrev = (e: React.MouseEvent) => { e.stopPropagation(); setPreviewIndex((i) => (i === null ? 0 : (i - 1 + total) % total)); };
          const goNext = (e: React.MouseEvent) => { e.stopPropagation(); setPreviewIndex((i) => (i === null ? 0 : (i + 1) % total)); };
          return (
            <div onClick={() => setPreviewIndex(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
              {/* Prev Button */}
              <button onClick={goPrev} style={{ position: 'absolute', left: 70, top: '50%', transform: 'translateY(-50%)', background: 'rgba(17,24,39,0.8)', color: '#fff', border: 10, borderRadius: '9999px', width: 50, height: 50, cursor: 'pointer' }}>{'<'}</button>
              {/* Next Button */}
              <button onClick={goNext} style={{ position: 'absolute', right: 70, top: '50%', transform: 'translateY(-50%)', background: 'rgba(17,24,39,0.8)', color: '#fff', border: 10, borderRadius: '9999px', width: 50, height: 50, cursor: 'pointer' }}>{'>'}</button>

              <div onClick={(e) => e.stopPropagation()} style={{ background: '#fff', borderRadius: 12, maxWidth: '90vw', maxHeight: '90vh', overflow: 'hidden', border: '1px solid #e5e7eb' }}>
                {imgSrc && (
                  <img src={imgSrc} alt={preview.image || ''} style={{ display: 'block', maxWidth: '90vw', maxHeight: '80vh', objectFit: 'contain' }} />
                )}
                <div style={{ padding: 12, fontSize: 14, color: '#111827', borderTop: '1px solid #e5e7eb', display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <div><b>Frame</b>: #{preview.frame_idx}</div>
                    <div><b>Score</b>: {(preview.score ?? 0).toFixed(4)}</div>
                    <div><b>Image</b>: {preview.image || '-'}</div>
                    {preview.frame_timestamp !== null && preview.frame_timestamp !== undefined && (
                      <div><b>Time</b>: {Math.floor(preview.frame_timestamp / 60)}:{(preview.frame_timestamp % 60).toFixed(1).padStart(4, '0')}</div>
                    )}
                    <div style={{ color: '#6b7280' }}>{preview.video_id}</div>
                    {preview.youtube_url && (
                      <button 
                        onClick={() => { setPreviewIndex(null); setVideoPreview(preview); }} 
                        style={{ 
                          display: 'inline-block',
                          padding: '4px 8px', 
                          background: '#ff0000', 
                          color: '#fff', 
                          border: 0,
                          borderRadius: 4, 
                          fontSize: 12,
                          fontWeight: 500,
                          cursor: 'pointer'
                        }}
                      >
                        📺 {preview.frame_timestamp !== null && preview.frame_timestamp !== undefined ? 'Preview Frame' : 'Preview Video'}
                      </button>
                    )}
                  </div>
                  <button onClick={() => setPreviewIndex(null)} style={{ padding: '6px 10px', background: '#111827', color: '#fff', border: 0, borderRadius: 6, cursor: 'pointer' }}>Close</button>
                </div>
              </div>
            </div>
          );
        })()}

        {/* Video Preview Modal */}
        {videoPreview && (() => {
          const vp = videoPreview;
          const embed = buildYouTubeEmbedUrl(vp.youtube_url, vp.frame_timestamp);
          return (
            <div onClick={() => setVideoPreview(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
              <div onClick={(e) => e.stopPropagation()} style={{ background: '#fff', borderRadius: 12, maxWidth: '90vw', maxHeight: '90vh', overflow: 'hidden', border: '1px solid #e5e7eb', width: 'min(960px, 90vw)' }}>
                <div style={{ position: 'relative', paddingBottom: '56.25%', height: 0 }}>
                  {embed ? (
                    <iframe
                      src={embed}
                      title="YouTube video player"
                      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0 }}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                      allowFullScreen
                    />
                  ) : (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>No YouTube URL available</div>
                  )}
                </div>
                <div style={{ padding: 12, fontSize: 14, color: '#111827', borderTop: '1px solid #e5e7eb', display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <div><b>Frame</b>: #{vp.frame_idx}</div>
                    {vp.frame_timestamp !== null && vp.frame_timestamp !== undefined && (
                      <div><b>Time</b>: {Math.floor(vp.frame_timestamp / 60)}:{(vp.frame_timestamp % 60).toFixed(1).padStart(4, '0')}</div>
                    )}
                    <div style={{ color: '#6b7280' }}>{vp.video_id}</div>
                  </div>
                  <button onClick={() => setVideoPreview(null)} style={{ padding: '6px 10px', background: '#111827', color: '#fff', border: 0, borderRadius: 6, cursor: 'pointer' }}>Close</button>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
};


