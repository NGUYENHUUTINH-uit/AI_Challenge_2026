import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import os
import json


# ===== CẤU HÌNH =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc dự án
VIDEO_DIR = os.path.join(BASE_DIR, "data", "sample_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "keyframes")
METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_keyframes(video_path: str, video_id: str, metadata_dict: dict):
    """Trích xuất 1 keyframe mỗi giây từ video và ghi metadata."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ❌ Không thể mở video: {video_path}")
        return 0

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        print(f"  ❌ FPS = 0, bỏ qua video: {video_path}")
        cap.release()
        return 0

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Trích xuất 1 frame mỗi giây
        if frame_count % fps == 0:
            timestamp = frame_count / fps

            # Tạo tên file: videoXXX_frame_0.jpg
            frame_name = f"{video_id}_frame_{saved_count}.jpg"
            frame_path = os.path.join(OUTPUT_DIR, frame_name)

            cv2.imwrite(frame_path, frame)

            metadata_dict[frame_name] = {
                "video_id": video_id,
                "timestamp": round(timestamp, 2)
            }

            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


# ===== CHẠY CHÍNH =====
# Đọc metadata cũ nếu có 
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
else:
    metadata_dict = {}

# Lấy danh sách tất cả video trong thư mục
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]

if not video_files:
    print(f"Không tìm thấy video nào trong '{VIDEO_DIR}'.")
else:
    print(f"Tìm thấy {len(video_files)} video. Bắt đầu xử lý...\n")
    total_frames = 0

    for i, filename in enumerate(video_files):
        video_id = os.path.splitext(filename)[0]  # Dùng tên file làm video_id (bỏ đuôi .mp4)
        video_path = os.path.join(VIDEO_DIR, filename)

        print(f"[{i+1}/{len(video_files)}] Đang xử lý: {filename}")
        saved = extract_keyframes(video_path, video_id, metadata_dict)
        print(f"  ✅ Đã lưu {saved} keyframe\n")
        total_frames += saved

    # Ghi metadata tổng hợp
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4, ensure_ascii=False)

    print(f"Hoàn tất! Tổng cộng {total_frames} keyframe từ {len(video_files)} video.")
    print(f"Metadata lưu tại: '{METADATA_PATH}'")