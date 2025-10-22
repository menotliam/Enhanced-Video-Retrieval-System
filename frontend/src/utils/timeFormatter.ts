/**
 * Format time in seconds to MM:SS or HH:MM:SS format
 * @param seconds - Time in seconds
 * @param showHours - Whether to show hours for times > 1 hour
 * @returns Formatted time string
 */
export const formatTime = (seconds: number, showHours: boolean = false): string => {
  if (seconds < 0) return '00:00';
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (showHours || hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

/**
 * Format duration in a human-readable format
 * @param seconds - Duration in seconds
 * @returns Human-readable duration string
 */
export const formatDuration = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.floor(seconds)} giây`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes} phút`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours} giờ ${minutes} phút`;
  }
};

/**
 * Format time range (start - end)
 * @param startTime - Start time in seconds
 * @param endTime - End time in seconds
 * @returns Formatted time range string
 */
export const formatTimeRange = (startTime: number, endTime: number): string => {
  const start = formatTime(startTime);
  const end = formatTime(endTime);
  return `${start} - ${end}`;
};

/**
 * Convert timestamp to relative time (e.g., "2 phút trước")
 * @param timestamp - Date timestamp
 * @returns Relative time string
 */
export const formatRelativeTime = (timestamp: Date | number): string => {
  const now = new Date();
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return 'Vừa xong';
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes} phút trước`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours} giờ trước`;
  } else if (diffInSeconds < 2592000) {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days} ngày trước`;
  } else if (diffInSeconds < 31536000) {
    const months = Math.floor(diffInSeconds / 2592000);
    return `${months} tháng trước`;
  } else {
    const years = Math.floor(diffInSeconds / 31536000);
    return `${years} năm trước`;
  }
};

/**
 * Format date in Vietnamese locale
 * @param date - Date to format
 * @param options - Intl.DateTimeFormatOptions
 * @returns Formatted date string
 */
export const formatDate = (
  date: Date | number,
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }
): string => {
  const dateObj = date instanceof Date ? date : new Date(date);
  return dateObj.toLocaleDateString('vi-VN', options);
};

/**
 * Format date and time in Vietnamese locale
 * @param date - Date to format
 * @returns Formatted date and time string
 */
export const formatDateTime = (date: Date | number): string => {
  const dateObj = date instanceof Date ? date : new Date(date);
  return dateObj.toLocaleString('vi-VN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
};

/**
 * Parse time string to seconds
 * @param timeString - Time string in format "MM:SS" or "HH:MM:SS"
 * @returns Time in seconds
 */
export const parseTimeString = (timeString: string): number => {
  const parts = timeString.split(':').map(Number);
  
  if (parts.length === 2) {
    // MM:SS format
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    // HH:MM:SS format
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  
  return 0;
};

/**
 * Get time segments for timeline display
 * @param duration - Total duration in seconds
 * @param segments - Number of segments to create
 * @returns Array of time points
 */
export const getTimeSegments = (duration: number, segments: number = 10): number[] => {
  const segmentDuration = duration / segments;
  return Array.from({ length: segments + 1 }, (_, i) => i * segmentDuration);
};

/**
 * Format search time in milliseconds
 * @param milliseconds - Search time in milliseconds
 * @returns Formatted search time string
 */
export const formatSearchTime = (milliseconds: number): string => {
  if (milliseconds < 1000) {
    return `${Math.round(milliseconds)}ms`;
  } else {
    return `${(milliseconds / 1000).toFixed(2)}s`;
  }
};

/**
 * Get time of day from timestamp
 * @param timestamp - Date timestamp
 * @returns Time of day string
 */
export const getTimeOfDay = (timestamp: Date | number): string => {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const hour = date.getHours();

  if (hour >= 5 && hour < 12) {
    return 'Sáng';
  } else if (hour >= 12 && hour < 17) {
    return 'Chiều';
  } else if (hour >= 17 && hour < 19) {
    return 'Tối';
  } else {
    return 'Đêm';
  }
};

/**
 * Check if time is within business hours
 * @param timestamp - Date timestamp
 * @returns Whether time is within business hours
 */
export const isBusinessHours = (timestamp: Date | number): boolean => {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const hour = date.getHours();
  const day = date.getDay();
  
  // Monday to Friday, 8 AM to 6 PM
  return day >= 1 && day <= 5 && hour >= 8 && hour < 18;
};

/**
 * Format video duration for display
 * @param duration - Duration in seconds
 * @returns Formatted duration string
 */
export const formatVideoDuration = (duration: number): string => {
  if (duration < 60) {
    return `${Math.floor(duration)}s`;
  } else if (duration < 3600) {
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  } else {
    const hours = Math.floor(duration / 3600);
    const minutes = Math.floor((duration % 3600) / 60);
    const seconds = Math.floor(duration % 60);
    return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }
};

/**
 * Get time progress percentage
 * @param currentTime - Current time in seconds
 * @param totalDuration - Total duration in seconds
 * @returns Progress percentage (0-100)
 */
export const getTimeProgress = (currentTime: number, totalDuration: number): number => {
  if (totalDuration <= 0) return 0;
  return Math.min(100, Math.max(0, (currentTime / totalDuration) * 100));
};

/**
 * Format time for accessibility
 * @param seconds - Time in seconds
 * @returns Accessible time string
 */
export const formatTimeForAccessibility = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];
  
  if (hours > 0) {
    parts.push(`${hours} giờ`);
  }
  if (minutes > 0) {
    parts.push(`${minutes} phút`);
  }
  if (secs > 0 || parts.length === 0) {
    parts.push(`${secs} giây`);
  }

  return parts.join(' ');
};
