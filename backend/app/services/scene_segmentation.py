def extract_and_save_frames(video_path: str, output_dir: str, img_ext: str = "jpg"):
    """
    Trích xuất và lưu toàn bộ frame từ video ra thư mục output_dir.
    Mỗi frame sẽ được lưu thành file ảnh (jpg/png).
    """
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video không tồn tại: {video_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"frame_{idx:05}.{img_ext}")
        cv2.imwrite(out_path, frame)
        idx += 1
    cap.release()
    print(f"[INFO] Đã lưu {idx} frame vào {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scene segmentation & extract frames from video")
    parser.add_argument(
        "--video",
        type=str,
        default="C:/Users/gbao2/OneDrive/Tài liệu/ai-video-search/VIDEO_AIC2024_P1/Videos_L01_a/video/L01_V001.mp4",
        help="Đường dẫn file video (.mp4), ví dụ: .../VIDEO_AIC2024_P1/Videos_L01_a/video/L01_V001.mp4"
    )
    parser.add_argument("--frames", type=str, help="Thư mục lưu frame (nếu muốn xuất frame)")
    parser.add_argument("--scenes", type=str, help="File JSON lưu kết quả phân cảnh (nếu muốn)")
    parser.add_argument(
        "--batch_folder",
        type=str,
        default="C:/Users/gbao2/OneDrive/Tài liệu/ai-video-search/VIDEO_AIC2024_P1",
        help="Phân cảnh toàn bộ video trong folder (nếu muốn), mặc định: VIDEO_AIC2024_P1 (tìm tất cả .mp4 trong các subfolder video/)"
    )
    args = parser.parse_args()

    if args.video and args.frames:
        extract_and_save_frames(args.video, args.frames)

    if args.video and args.scenes:
        try:
            from scene_segmentation import SceneSegmenter
            print("[DEBUG] Bắt đầu phân cảnh video...")
            segmenter = SceneSegmenter()
            scenes = segmenter.segment(args.video)
            print("[DEBUG] Bắt đầu ghi file output...")
            with open(args.scenes, "w", encoding="utf-8") as f:
                json.dump(scenes, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Đã lưu kết quả phân cảnh vào {args.scenes}")
        except Exception as e:
            print(f"[ERROR] {e}")

    if args.batch_folder and args.scenes:
        try:
            from scene_segmentation import SceneSegmenter
            print("[DEBUG] Bắt đầu phân cảnh batch...")
            segmenter = SceneSegmenter()
            segmenter.segment_videos_in_folder(args.batch_folder, output_json=args.scenes)
        except Exception as e:
            print(f"[ERROR] {e}")

"""
Scene segmentation using TransNetV2.
Input: Video file path hoặc thư mục chứa nhiều video
Output: List of scenes với (scene_id, start_ts, end_ts) cho từng video, lưu ra file JSON nếu cần
"""

from typing import List, Dict

import os
import numpy as np
import cv2
from typing import List, Dict


import json
from glob import glob

class SceneSegmenter:
    def __init__(self, model_path: str = None):
        """
        Khởi tạo TransNetV2 model.
        model_path: đường dẫn model nếu muốn custom, mặc định dùng pretrained.
        """
        try:
            from transnetv2 import TransNetV2
        except ImportError:
            raise ImportError("You need to install transnetv2: pip install transnetv2")
        self.model = TransNetV2(model_path) if model_path else TransNetV2()


    def segment(self, video_path: str) -> List[Dict]:
        """
        Phân cảnh 1 video, trả về list dict (scene_id, start_ts, end_ts)
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        scenes = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames == 0 or fps == 0:
            raise ValueError(f"Cannot read video or video is empty: {video_path}")
        frames = self._extract_frames(video_path)
        predictions = self.model.predict_frames(frames)
        scene_list = self._get_scenes_from_predictions(predictions, fps, total_frames)
        for idx, (start, end) in enumerate(scene_list):
            scenes.append({
                "scene_id": f"scene_{idx+1}",
                "start_ts": self._frame_to_timecode(start, fps),
                "end_ts": self._frame_to_timecode(end, fps)
            })
        return scenes

    def segment_videos_in_folder(self, folder_path: str, output_json: str = None, recursive: bool = True) -> Dict[str, List[Dict]]:
        """
        Phân cảnh toàn bộ video mp4 trong folder (tìm tất cả file .mp4 trong các subfolder video/).
        output_json: nếu truyền vào, lưu kết quả ra file JSON.
        Return: dict {video_path: [scene_dict, ...]}
        """
        # Tìm tất cả file .mp4 trong các subfolder video/ bên trong folder_path
        pattern = os.path.join(folder_path, "Videos_*_a", "video", "*.mp4")
        video_files = glob(pattern, recursive=False)
        results = {}
        for video in video_files:
            try:
                scenes = self.segment(video)
                results[video] = scenes
                print(f"[OK] {video}: {len(scenes)} scenes")
            except Exception as e:
                print(f"[ERROR] {video}: {e}")
        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    def _extract_frames(self, video_path: str) -> np.ndarray:
        """
        Extract all frames from video as numpy array (N, H, W, 3).
        Cảnh báo: Tốn RAM nếu video dài.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        return np.array(frames)

    def _get_scenes_from_predictions(self, predictions, fps, total_frames):
        """
        Convert TransNetV2 predictions to list of (start_frame, end_frame).
        """
        scenes = []
        boundaries = np.where(predictions > 0.5)[0]
        if len(boundaries) == 0:
            return [(0, total_frames-1)]
        prev = 0
        for b in boundaries:
            scenes.append((prev, b))
            prev = b+1
        if prev < total_frames:
            scenes.append((prev, total_frames-1))
        return scenes

    def _frame_to_timecode(self, frame_idx, fps):
        """Convert frame index to HH:MM:SS.mmm timecode."""
        seconds = frame_idx / fps
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
