import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Play, 
  Clock, 
  Tag, 
  MapPin, 
  Star, 
  Eye, 
  Volume2,
  Maximize2,
  Share2,
  Bookmark,
  BookmarkPlus
} from 'lucide-react';
import { Scene } from '@/types';
import { useSearchStore } from '@/store';
import { formatTime } from '@/utils/timeFormatter';
import { highlightVietnameseText } from '@/utils/vnTextHighlight';

interface ScenePreviewProps {
  scene: Scene;
  onClick: (scene: Scene) => void;
  viewMode: 'grid' | 'list';
  highlightTerms?: string[];
  index: number;
}

export const ScenePreview: React.FC<ScenePreviewProps> = ({
  scene,
  onClick,
  viewMode,
  highlightTerms = [],
  index
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [showFullDescription, setShowFullDescription] = useState(false);
  
  const setSelectedScene = useSearchStore(state => state.setSelectedScene);

  const handleClick = useCallback(() => {
    setSelectedScene(scene);
    onClick(scene);
  }, [scene, onClick, setSelectedScene]);

  const handleBookmark = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsBookmarked(!isBookmarked);
  }, [isBookmarked]);

  const handleShare = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // Share functionality
    if (navigator.share) {
      navigator.share({
        title: `Scene from ${scene.video_id}`,
        text: scene.scene_description,
        url: window.location.href
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(
        `Scene: ${scene.scene_description}\nTime: ${formatTime(scene.start_time)} - ${formatTime(scene.end_time)}`
      );
    }
  }, [scene]);

  const handleExpand = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // Open in full screen modal
    setSelectedScene(scene);
  }, [scene, setSelectedScene]);

  // Calculate quality score
  const qualityScore = Math.round(
    (scene.metadata.cross_encoder_score + 
     scene.metadata.fuzzy_score + 
     scene.metadata.quality_score) / 3 * 100
  );

  // Highlight text in Vietnamese
  const highlightedTranscript = highlightVietnameseText(scene.transcript, highlightTerms);
  const highlightedOCR = highlightVietnameseText(scene.ocr_text, highlightTerms);

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { delay: index * 0.1 }
    },
    hover: { 
      y: -4,
      transition: { duration: 0.2 }
    }
  };

  if (viewMode === 'list') {
    return (
      <motion.div
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        whileHover="hover"
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        onClick={handleClick}
        className="bg-white rounded-xl border border-gray-200 hover:border-primary-300 hover:shadow-lg transition-all cursor-pointer overflow-hidden"
      >
        <div className="flex">
          {/* Thumbnail */}
          <div className="relative w-48 h-32 flex-shrink-0">
            <img
              src={scene.preview_image_url}
              alt={scene.scene_description}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-20 transition-all flex items-center justify-center">
              <Play className="text-white opacity-0 group-hover:opacity-100 transition-opacity" size={24} />
            </div>
            <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
              {formatTime(scene.start_time)}
            </div>
            <div className="absolute top-2 right-2 bg-primary-600 text-white text-xs px-2 py-1 rounded flex items-center">
              <Star size={12} className="mr-1" />
              {qualityScore}%
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-1 line-clamp-2">
                  {scene.scene_description}
                </h3>
                <div className="flex items-center text-sm text-gray-500 mb-2">
                  <Clock size={14} className="mr-1" />
                  {formatTime(scene.start_time)} - {formatTime(scene.end_time)}
                  <span className="mx-2">•</span>
                  <span>{scene.duration}s</span>
                </div>
              </div>
              
              {/* Action buttons */}
              <div className="flex items-center space-x-1 ml-4">
                <button
                  onClick={handleBookmark}
                  className={`p-2 rounded-lg transition-colors ${
                    isBookmarked 
                      ? 'bg-primary-100 text-primary-600' 
                      : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {isBookmarked ? <Bookmark size={16} /> : <BookmarkPlus size={16} />}
                </button>
                <button
                  onClick={handleShare}
                  className="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                >
                  <Share2 size={16} />
                </button>
                <button
                  onClick={handleExpand}
                  className="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                >
                  <Maximize2 size={16} />
                </button>
              </div>
            </div>

            {/* Transcript */}
            <div className="mb-2">
              <div className="text-sm text-gray-700 line-clamp-2">
                <span className="font-medium">Transcript:</span> {highlightedTranscript}
              </div>
            </div>

            {/* OCR Text */}
            {scene.ocr_text && (
              <div className="mb-2">
                <div className="text-sm text-gray-700 line-clamp-1">
                  <span className="font-medium">OCR:</span> {highlightedOCR}
                </div>
              </div>
            )}

            {/* Objects and Metadata */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {scene.detected_objects.slice(0, 3).map((object, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-700"
                  >
                    <Tag size={10} className="mr-1" />
                    {object}
                  </span>
                ))}
                {scene.detected_objects.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{scene.detected_objects.length - 3} more
                  </span>
                )}
              </div>

              {scene.metadata.location && (
                <div className="flex items-center text-xs text-gray-500">
                  <MapPin size={12} className="mr-1" />
                  {scene.metadata.location}
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  // Grid view
  return (
    <motion.div
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      whileHover="hover"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      onClick={handleClick}
      className="bg-white rounded-xl border border-gray-200 hover:border-primary-300 hover:shadow-lg transition-all cursor-pointer overflow-hidden group"
    >
      {/* Thumbnail */}
      <div className="relative aspect-video">
        <img
          src={scene.preview_image_url}
          alt={scene.scene_description}
          className="w-full h-full object-cover"
        />
        
        {/* Overlay */}
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all flex items-center justify-center">
          <Play className="text-white opacity-0 group-hover:opacity-100 transition-opacity" size={32} />
        </div>

        {/* Time badge */}
        <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
          {formatTime(scene.start_time)}
        </div>

        {/* Quality score */}
        <div className="absolute top-2 right-2 bg-primary-600 text-white text-xs px-2 py-1 rounded flex items-center">
          <Star size={12} className="mr-1" />
          {qualityScore}%
        </div>

        {/* Action buttons */}
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="flex space-x-1">
            <button
              onClick={handleBookmark}
              className={`p-1.5 rounded-lg transition-colors ${
                isBookmarked 
                  ? 'bg-primary-100 text-primary-600' 
                  : 'bg-black bg-opacity-75 text-white hover:bg-opacity-90'
              }`}
            >
              {isBookmarked ? <Bookmark size={14} /> : <BookmarkPlus size={14} />}
            </button>
            <button
              onClick={handleShare}
              className="p-1.5 rounded-lg bg-black bg-opacity-75 text-white hover:bg-opacity-90 transition-colors"
            >
              <Share2 size={14} />
            </button>
            <button
              onClick={handleExpand}
              className="p-1.5 rounded-lg bg-black bg-opacity-75 text-white hover:bg-opacity-90 transition-colors"
            >
              <Maximize2 size={14} />
            </button>
          </div>
        </div>

        {/* Audio indicator */}
        {scene.audio_snippet_url && (
          <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white p-1 rounded">
            <Volume2 size={12} />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Title and duration */}
        <div className="flex items-start justify-between mb-2">
          <h3 className="font-semibold text-gray-900 text-sm line-clamp-2 flex-1">
            {scene.scene_description}
          </h3>
          <div className="flex items-center text-xs text-gray-500 ml-2 flex-shrink-0">
            <Clock size={12} className="mr-1" />
            {scene.duration}s
          </div>
        </div>

        {/* Transcript preview */}
        <div className="mb-3">
          <div className="text-xs text-gray-700 line-clamp-2">
            {highlightedTranscript}
          </div>
        </div>

        {/* Objects */}
        <div className="flex flex-wrap gap-1 mb-2">
          {scene.detected_objects.slice(0, 2).map((object, idx) => (
            <span
              key={idx}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-700"
            >
              <Tag size={10} className="mr-1" />
              {object}
            </span>
          ))}
          {scene.detected_objects.length > 2 && (
            <span className="text-xs text-gray-500 self-center">
              +{scene.detected_objects.length - 2}
            </span>
          )}
        </div>

        {/* Metadata */}
        <div className="flex items-center justify-between text-xs text-gray-500">
          {scene.metadata.location && (
            <div className="flex items-center">
              <MapPin size={10} className="mr-1" />
              {scene.metadata.location}
            </div>
          )}
          <div className="flex items-center">
            <Eye size={10} className="mr-1" />
            Score: {Math.round(scene.score * 100)}%
          </div>
        </div>
      </div>
    </motion.div>
  );
};
