import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  SkipBack, 
  SkipForward, 
  Volume2, 
  VolumeX, 
  Maximize2, 
  Minimize2,
  Settings,
  Download,
  Share2,
  X,
  Clock,
  Tag,
  MapPin,
  Star,
  Eye,
  Copy,
  ExternalLink
} from 'lucide-react';
import { Scene } from '@/types';
import { useSearchStore } from '@/store';
import { formatTime } from '@/utils/timeFormatter';
import { highlightVietnameseText } from '@/utils/vnTextHighlight';

interface VideoPlayerProps {
  scene: Scene;
  autoPlay?: boolean;
  onClose: () => void;
  onSceneChange?: (scene: Scene) => void;
  showFullscreen?: boolean;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  scene,
  autoPlay = true,
  onClose,
  onSceneChange,
  showFullscreen = false
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(showFullscreen);
  const [showControls, setShowControls] = useState(true);
  const [showSubtitles, setShowSubtitles] = useState(true);
  const [showMetadata, setShowMetadata] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const controlsTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const setSelectedScene = useSearchStore(state => state.setSelectedScene);

  // Initialize video
  useEffect(() => {
    if (videoRef.current) {
      const video = videoRef.current;
      
      // Set video source
      video.src = scene.audio_snippet_url || '';
      video.currentTime = scene.start_time;
      
      // Event listeners
      const handleLoadedMetadata = () => {
        setDuration(video.duration);
        if (autoPlay) {
          video.play().catch(console.error);
        }
      };

      const handleTimeUpdate = () => {
        setCurrentTime(video.currentTime);
      };

      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);
      const handleWaiting = () => setBuffering(true);
      const handleCanPlay = () => setBuffering(false);
      const handleError = () => setError('Không thể phát video');

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);
      video.addEventListener('waiting', handleWaiting);
      video.addEventListener('canplay', handleCanPlay);
      video.addEventListener('error', handleError);

      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
        video.removeEventListener('waiting', handleWaiting);
        video.removeEventListener('canplay', handleCanPlay);
        video.removeEventListener('error', handleError);
      };
    }
  }, [scene, autoPlay]);

  // Auto-hide controls
  useEffect(() => {
    if (showControls) {
      controlsTimeoutRef.current = setTimeout(() => {
        setShowControls(false);
      }, 3000);
    }

    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current);
      }
    };
  }, [showControls]);

  // Fullscreen handling
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Playback controls
  const togglePlay = useCallback(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play().catch(console.error);
      }
    }
  }, [isPlaying]);

  const seekTo = useCallback((time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  }, []);

  const skipBackward = useCallback(() => {
    seekTo(Math.max(0, currentTime - 10));
  }, [currentTime, seekTo]);

  const skipForward = useCallback(() => {
    seekTo(Math.min(duration, currentTime + 10));
  }, [currentTime, duration, seekTo]);

  const toggleMute = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  }, [isMuted]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
    }
  }, []);

  const toggleFullscreen = useCallback(async () => {
    if (containerRef.current) {
      if (isFullscreen) {
        await document.exitFullscreen();
      } else {
        await containerRef.current.requestFullscreen();
      }
    }
  }, [isFullscreen]);

  const handleMouseMove = useCallback(() => {
    setShowControls(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setShowControls(false);
  }, []);

  // Utility functions
  const copySceneInfo = useCallback(() => {
    const info = `Scene: ${scene.scene_description}
Time: ${formatTime(scene.start_time)} - ${formatTime(scene.end_time)}
Transcript: ${scene.transcript}
Score: ${Math.round(scene.score * 100)}%`;
    
    navigator.clipboard.writeText(info);
  }, [scene]);

  const shareScene = useCallback(() => {
    if (navigator.share) {
      navigator.share({
        title: `Scene from ${scene.video_id}`,
        text: scene.scene_description,
        url: window.location.href
      });
    } else {
      copySceneInfo();
    }
  }, [scene, copySceneInfo]);

  // Calculate progress
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
  const sceneProgress = duration > 0 ? 
    ((currentTime - scene.start_time) / (scene.end_time - scene.start_time)) * 100 : 0;

  // Quality score
  const qualityScore = Math.round(
    (scene.metadata.cross_encoder_score + 
     scene.metadata.fuzzy_score + 
     scene.metadata.quality_score) / 3 * 100
  );

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          ref={containerRef}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-6xl bg-black rounded-xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Video Element */}
          <video
            ref={videoRef}
            className="w-full h-full object-contain"
            poster={scene.preview_image_url}
            preload="metadata"
          />

          {/* Buffering Indicator */}
          {buffering && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
              <div className="text-white text-center">
                <div className="text-2xl mb-2">⚠️</div>
                <div className="text-lg mb-2">{error}</div>
                <button
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                >
                  Thử lại
                </button>
              </div>
            </div>
          )}

          {/* Subtitles */}
          {showSubtitles && scene.transcript && (
            <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-75 text-white px-4 py-2 rounded-lg max-w-2xl text-center">
              <div className="text-sm">
                {highlightVietnameseText(scene.transcript, [])}
              </div>
            </div>
          )}

          {/* Controls Overlay */}
          <AnimatePresence>
            {showControls && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent"
              >
                {/* Top Controls */}
                <div className="absolute top-0 left-0 right-0 p-4 flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <h2 className="text-white font-semibold text-lg truncate">
                      {scene.scene_description}
                    </h2>
                    <div className="flex items-center text-white text-sm">
                      <Star size={14} className="mr-1" />
                      {qualityScore}%
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setShowMetadata(!showMetadata)}
                      className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
                    >
                      <Settings size={20} />
                    </button>
                    <button
                      onClick={shareScene}
                      className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
                    >
                      <Share2 size={20} />
                    </button>
                    <button
                      onClick={toggleFullscreen}
                      className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
                    >
                      {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
                    </button>
                    <button
                      onClick={onClose}
                      className="p-2 text-white hover:bg-white hover:bg-opacity-20 rounded-lg transition-colors"
                    >
                      <X size={20} />
                    </button>
                  </div>
                </div>

                {/* Center Play Button */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <button
                    onClick={togglePlay}
                    className="p-4 bg-white bg-opacity-20 rounded-full hover:bg-opacity-30 transition-all"
                  >
                    {isPlaying ? <Pause size={32} /> : <Play size={32} />}
                  </button>
                </div>

                {/* Bottom Controls */}
                <div className="absolute bottom-0 left-0 right-0 p-4">
                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="relative h-1 bg-white bg-opacity-30 rounded-full">
                      <div 
                        className="absolute h-full bg-primary-500 rounded-full"
                        style={{ width: `${progress}%` }}
                      />
                      <div 
                        className="absolute h-full bg-yellow-400 rounded-full"
                        style={{ 
                          width: `${sceneProgress}%`,
                          left: `${(scene.start_time / duration) * 100}%`
                        }}
                      />
                    </div>
                  </div>

                  {/* Control Buttons */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <button
                        onClick={togglePlay}
                        className="text-white hover:text-primary-400 transition-colors"
                      >
                        {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                      </button>
                      
                      <button
                        onClick={skipBackward}
                        className="text-white hover:text-primary-400 transition-colors"
                      >
                        <SkipBack size={20} />
                      </button>
                      
                      <button
                        onClick={skipForward}
                        className="text-white hover:text-primary-400 transition-colors"
                      >
                        <SkipForward size={20} />
                      </button>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={toggleMute}
                          className="text-white hover:text-primary-400 transition-colors"
                        >
                          {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
                        </button>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={volume}
                          onChange={handleVolumeChange}
                          className="w-20"
                        />
                      </div>

                      <div className="text-white text-sm">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setShowSubtitles(!showSubtitles)}
                        className={`px-3 py-1 rounded text-sm transition-colors ${
                          showSubtitles 
                            ? 'bg-primary-600 text-white' 
                            : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30'
                        }`}
                      >
                        CC
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Metadata Panel */}
          <AnimatePresence>
            {showMetadata && (
              <motion.div
                initial={{ x: '100%' }}
                animate={{ x: 0 }}
                exit={{ x: '100%' }}
                className="absolute top-0 right-0 w-80 h-full bg-white bg-opacity-95 backdrop-blur-sm overflow-y-auto"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">Thông tin Scene</h3>
                    <button
                      onClick={() => setShowMetadata(false)}
                      className="p-1 hover:bg-gray-200 rounded"
                    >
                      <X size={16} />
                    </button>
                  </div>

                  {/* Scene Info */}
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Mô tả</h4>
                      <p className="text-sm text-gray-700">{scene.scene_description}</p>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Thời gian</h4>
                      <div className="flex items-center text-sm text-gray-700">
                        <Clock size={14} className="mr-2" />
                        {formatTime(scene.start_time)} - {formatTime(scene.end_time)} ({scene.duration}s)
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Transcript</h4>
                      <p className="text-sm text-gray-700">{scene.transcript}</p>
                    </div>

                    {scene.ocr_text && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">OCR Text</h4>
                        <p className="text-sm text-gray-700">{scene.ocr_text}</p>
                      </div>
                    )}

                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Đối tượng phát hiện</h4>
                      <div className="flex flex-wrap gap-1">
                        {scene.detected_objects.map((object, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-700"
                          >
                            <Tag size={10} className="mr-1" />
                            {object}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Điểm số</h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Cross-encoder:</span>
                          <span>{Math.round(scene.metadata.cross_encoder_score * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span>Fuzzy match:</span>
                          <span>{Math.round(scene.metadata.fuzzy_score * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span>Quality:</span>
                          <span>{Math.round(scene.metadata.quality_score * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between text-sm font-medium">
                          <span>Tổng cộng:</span>
                          <span>{Math.round(scene.score * 100)}%</span>
                        </div>
                      </div>
                    </div>

                    {scene.metadata.location && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Vị trí</h4>
                        <div className="flex items-center text-sm text-gray-700">
                          <MapPin size={14} className="mr-2" />
                          {scene.metadata.location}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="pt-4 border-t border-gray-200">
                      <div className="flex space-x-2">
                        <button
                          onClick={copySceneInfo}
                          className="flex-1 flex items-center justify-center px-3 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                        >
                          <Copy size={14} className="mr-2" />
                          Sao chép
                        </button>
                        <button
                          onClick={shareScene}
                          className="flex-1 flex items-center justify-center px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                        >
                          <Share2 size={14} className="mr-2" />
                          Chia sẻ
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};
