"""
run_ocr_worker.py — Chạy OCR phân tán cho nhóm (Tự động lưu Google Drive Real-time)
===================================================================================
Cách dùng:
  1. Chạy theo Lesson (Khuyên dùng):
     Người 1:  python -m src.indexing.run_ocr_worker --lessons L21,L22,L23,L24,L27,L28,L29,L30
     Người 2:  python -m src.indexing.run_ocr_worker --lessons L25
     Người 3:  python -m src.indexing.run_ocr_worker --lessons L26

  2. Gộp tất cả sau khi chạy xong:
     python -m src.indexing.run_ocr_worker --merge

Tính năng nổi bật:
  - TỰ ĐỘNG SYNC VÀO GOOGLE DRIVE: Cứ sau mỗi 1 video hoàn thành, file JSON được ghi thẳng vào Google Drive.
  - TỰ ĐỘNG RESUME: Nếu bị ngắt kết nối/đổi máy ảo, script tự nạp lại tiến độ từ Google Drive và chạy tiếp.
  - SIÊU NHẸ: In log 1 dòng/video, không gây giật lag trình duyệt.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BTC_KEYFRAMES = PROJECT_ROOT / "data" / "btc" / "Keyframes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OCR_INDEX_FILE = PROCESSED_DIR / "ocr_index.json"

# Đường dẫn tự động nhận diện Google Drive trên Colab
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/AIC2026/ocr_results")


# ──────────────────────────────────────────────
# 1. CHIA DANH SÁCH VIDEO
# ──────────────────────────────────────────────

def get_all_video_dirs() -> list:
    """Lấy toàn bộ thư mục video, sort để đảm bảo thứ tự nhất quán."""
    if not BTC_KEYFRAMES.exists():
        return []
    dirs = sorted([d for d in BTC_KEYFRAMES.iterdir() if d.is_dir()])
    return dirs


def get_videos_by_lessons(lessons_str: str) -> tuple[list, str]:
    """
    Lấy danh sách video theo lesson prefix (ví dụ: 'L21,L22' hoặc 'L25').
    Trả về: (danh sách video dirs, tag tên file)
    """
    all_videos = get_all_video_dirs()
    prefixes = [l.strip().upper() for l in lessons_str.split(",") if l.strip()]
    
    assigned = [v for v in all_videos if any(v.name.upper().startswith(p) for p in prefixes)]
    tag = "_".join(prefixes)
    print(f"[OCR Worker] Chon cac Lesson: {prefixes} -> {len(assigned)}/{len(all_videos)} video")
    return assigned, tag


def get_videos_by_part(part: int, total: int) -> tuple[list, str]:
    """Trả về danh sách video folder theo phần (Part X of Total)."""
    all_videos = get_all_video_dirs()
    n = len(all_videos)
    chunk_size = (n + total - 1) // total
    start = (part - 1) * chunk_size
    end = min(start + chunk_size, n)
    assigned = all_videos[start:end]
    tag = f"{part}_of_{total}"
    print(f"[OCR Worker] Phan {part}/{total}: video #{start+1} -> #{end} "
          f"({len(assigned)} video, tong {n} video)")
    return assigned, tag


# ──────────────────────────────────────────────
# 2. LOAD OCR ENGINE
# ──────────────────────────────────────────────

def load_easyocr(use_gpu: bool = True):
    """Load EasyOCR reader. Hỗ trợ tiếng Việt + tiếng Anh."""
    try:
        import easyocr
        print(f"[OCR] Loading EasyOCR (GPU={'on' if use_gpu else 'off'})...")
        reader = easyocr.Reader(['vi', 'en'], gpu=use_gpu, verbose=False)
        print("[OCR] EasyOCR OK")
        return reader
    except ImportError:
        print("[OCR] EasyOCR chua cai. Chay: pip install easyocr")
        raise


# ──────────────────────────────────────────────
# 3. OCR MỘT ẢNH
# ──────────────────────────────────────────────

def ocr_image(reader, img_path: Path) -> str:
    """Chạy EasyOCR trên 1 ảnh, trả về chuỗi text đã ghép."""
    try:
        results = reader.readtext(str(img_path), detail=0, paragraph=True)
        text = " ".join(results).strip()
        return text
    except Exception:
        return ""


# ──────────────────────────────────────────────
# 4. PIPELINE CHÍNH (REAL-TIME GOOGLE DRIVE SYNC)
# ──────────────────────────────────────────────

def run_ocr(target_videos: list, tag: str, use_gpu: bool = True):
    """
    Chạy OCR cho danh sách video được phân công.
    - Tự động nạp tiến độ cũ từ Drive / Local để Resume.
    - Cứ xong 1 video là GHI TRỰC TIẾP VÀO GOOGLE DRIVE.
    - Không bao giờ mất dữ liệu kể cả khi máy ảo Colab bị restart.
    """
    if not target_videos:
        print("[OCR] Khong tim thay video nao phu hop de chay OCR!")
        return

    output_file = PROCESSED_DIR / f"ocr_part_{tag}.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    drive_dir = DEFAULT_DRIVE_DIR
    drive_file = drive_dir / f"ocr_part_{tag}.json" if drive_dir.parent.exists() else None
    if drive_file:
        drive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Khôi phục tiến độ cũ (Ưu tiên đọc từ Drive trước, rồi đến Local)
    ocr_data: dict = {}
    if drive_file and drive_file.exists():
        try:
            with open(drive_file, encoding="utf-8") as f:
                ocr_data = json.load(f)
            print(f"[Resume] Da tim thay va nap {len(ocr_data)} video da xong tu Google Drive!")
        except Exception:
            pass

    if not ocr_data and output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                ocr_data = json.load(f)
            print(f"[Resume] Da nap {len(ocr_data)} video da xong tu o dia local.")
        except Exception:
            ocr_data = {}

    # Load EasyOCR
    reader = load_easyocr(use_gpu=use_gpu)

    # Danh sách video cần xử lý
    pending = [v for v in target_videos if v.name not in ocr_data]
    print(f"[OCR Worker] Tong video: {len(target_videos)} | Da xong: {len(ocr_data)} | Con lai: {len(pending)}")

    total_imgs = sum(len(list(v.glob("*.jpg"))) for v in pending)
    print(f"[OCR Worker] Tong so anh can OCR tiep theo: {total_imgs:,}")

    t0 = time.time()
    processed_imgs = 0
    total_pending_vids = len(pending)

    for idx, v_dir in enumerate(pending, start=1):
        t_v0 = time.time()
        video_id = v_dir.name
        img_files = sorted(v_dir.glob("*.jpg"))

        video_ocr: dict = {}
        for img_path in img_files:
            text = ocr_image(reader, img_path)
            if text:
                video_ocr[img_path.name] = text
            processed_imgs += 1

        # 1. Lưu vào file local
        ocr_data[video_id] = video_ocr
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

        # 2. ĐỒNG BỘ TRỰC TIẾP VÀO GOOGLE DRIVE NGAY LẬP TỨC
        if drive_file:
            try:
                shutil.copy2(output_file, drive_file)
            except Exception:
                pass

        v_elapsed = time.time() - t_v0
        speed = len(img_files) / max(v_elapsed, 0.001)
        print(f"[{idx}/{total_pending_vids}] {video_id}: {len(img_files)} anh ({speed:.1f} anh/s, {v_elapsed:.1f}s) | Co text: {len(video_ocr)} | Tong: {len(ocr_data)}/{len(target_videos)} video da luu Drive")

    elapsed = time.time() - t0
    has_text = sum(len(v) for v in ocr_data.values())

    print(f"\n{'='*55}")
    print(f"  OCR WORKER [{tag}] - HOAN TAT 100%!")
    print(f"{'='*55}")
    print(f"  Tong video   : {len(target_videos)}")
    print(f"  Anh da OCR   : {processed_imgs:,}")
    print(f"  Anh co text  : {has_text:,}")
    print(f"  Thoi gian    : {elapsed/60:.1f} phut")
    if drive_file:
        print(f"  File tren Drive: {drive_file}")
    print(f"  File tren Local: {output_file}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────
# 5. MERGE TẤT CẢ FILE PARTIAL
# ──────────────────────────────────────────────

def merge_all_parts():
    """
    Gom toàn bộ file ocr_part_*.json thành 1 file ocr_index.json.
    Tự động quét cả thư mục local và Google Drive.
    """
    # 1. Quét tìm tất cả các file partial
    part_files_dict: dict[str, Path] = {}

    # Quét thư mục local
    for f in PROCESSED_DIR.glob("ocr_part_*.json"):
        part_files_dict[f.name] = f

    # Quét thư mục Google Drive (nếu có)
    if DEFAULT_DRIVE_DIR.exists():
        for f in DEFAULT_DRIVE_DIR.glob("ocr_part_*.json"):
            part_files_dict[f.name] = f

    if not part_files_dict:
        print("[Merge] Khong tim thay file ocr_part_*.json trong data/processed/ hoac Google Drive!")
        return

    print(f"[Merge] Tim thay {len(part_files_dict)} file partial:")
    for name, f in part_files_dict.items():
        size_kb = f.stat().st_size / 1024
        print(f"  - {name} ({size_kb:.0f} KB) tu {f.parent}")

    merged: dict = {}
    total_videos = 0
    total_frames = 0

    for name, part_file in part_files_dict.items():
        try:
            with open(part_file, encoding="utf-8") as f:
                data = json.load(f)
            for video_id, frames in data.items():
                merged[video_id] = frames
                total_videos += 1
                total_frames += len(frames)
            n_frames = sum(len(v) for v in data.values())
            print(f"  [OK] Da nap {name}: {len(data)} video, {n_frames} frame co OCR")
        except Exception as e:
            print(f"  [WARN] Loi doc file {name}: {e}")

    with open(OCR_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Nếu có Google Drive thì lưu luôn 1 bản vào Drive
    if DEFAULT_DRIVE_DIR.exists():
        try:
            shutil.copy2(OCR_INDEX_FILE, DEFAULT_DRIVE_DIR / "ocr_index.json")
            print(f"  [Drive] Da sao luu ocr_index.json len Drive: {DEFAULT_DRIVE_DIR / 'ocr_index.json'}")
        except Exception:
            pass

    size_mb = OCR_INDEX_FILE.stat().st_size / 1024 / 1024
    print(f"\n{'='*55}")
    print(f"  MERGE HOAN TAT!")
    print(f"{'='*55}")
    print(f"  Tong video   : {len(merged)}")
    print(f"  Frame co text: {total_frames:,}")
    print(f"  File output  : {OCR_INDEX_FILE}")
    print(f"  Kich thuoc   : {size_mb:.1f} MB")
    print(f"{'='*55}")


# ──────────────────────────────────────────────
# 6. THỐNG KÊ CHI TIẾT TỪNG LESSON
# ──────────────────────────────────────────────

def show_lesson_summary():
    """In ra danh sách chi tiết các Lesson và thời gian chạy ước tính."""
    from collections import defaultdict
    all_videos = get_all_video_dirs()
    if not all_videos:
        print("[WARN] Khong tim thay thu muc keyframes trong data/btc/Keyframes")
        return

    lesson_counts = defaultdict(lambda: {'videos': 0, 'imgs': 0})
    for v in all_videos:
        l_name = v.name[:3].upper()
        imgs = len(list(v.glob("*.jpg")))
        lesson_counts[l_name]['videos'] += 1
        lesson_counts[l_name]['imgs'] += imgs

    print(f"\n{'='*62}")
    print(f"  THONG KE CHI TIET THEO LESSON (Tong: {len(all_videos)} video)")
    print(f"{'='*62}")
    print(f"{'Lesson':<8} {'Videos':>8} {'So anh':>10} {'GPU T4 (~0.4s)':>18} {'CPU (~1.5s)':>12}")
    print("-" * 62)

    total_v, total_i = 0, 0
    for l_name, d in sorted(lesson_counts.items()):
        v, i = d['videos'], d['imgs']
        gpu_min = i * 0.4 / 60
        cpu_h = i * 1.5 / 3600
        if gpu_min >= 60:
            gpu_str = f"{gpu_min/60:.1f}h"
        else:
            gpu_str = f"{gpu_min:.0f} phut"
        print(f"{l_name:<8} {v:>8} {i:>10,} {gpu_str:>18} {cpu_h:>11.1f}h")
        total_v += v
        total_i += i

    print("-" * 62)
    print(f"{'TONG':<8} {total_v:>8} {total_i:>10,} {total_i*0.4/3600:>17.1f}h {total_i*1.5/3600:>11.1f}h")
    print(f"{'='*62}\n")
    print("Goi y lenh chay:")
    print("  Nguoi 1: python -m src.indexing.run_ocr_worker --lessons L21,L22,L23,L24,L27,L28,L29,L30")
    print("  Nguoi 2: python -m src.indexing.run_ocr_worker --lessons L25")
    print("  Nguoi 3: python -m src.indexing.run_ocr_worker --lessons L26\n")


# ──────────────────────────────────────────────
# 7. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Worker - Chay OCR voi tinh nang Real-time Drive Sync",
    )
    parser.add_argument("--lessons", type=str, default=None,
                        help="Danh sach cac lesson can chay (vi du: 'L21,L22' hoac 'L25')")
    parser.add_argument("--part", type=int, default=None,
                        help="So thu tu phan (neu dung che do chia deu part)")
    parser.add_argument("--total", type=int, default=3,
                        help="Tong so phan chia deu")
    parser.add_argument("--merge", action="store_true",
                        help="Merge tat ca file ocr_part_*.json thanh ocr_index.json")
    parser.add_argument("--plan", action="store_true",
                        help="Xem thong ke chi tiet theo Lesson va ke hoach goi y")
    parser.add_argument("--cpu", action="store_true",
                        help="Ep dung CPU thay vi GPU (cham hon)")
    args = parser.parse_args()

    if args.merge:
        merge_all_parts()
    elif args.plan:
        show_lesson_summary()
    elif args.lessons:
        videos, tag = get_videos_by_lessons(args.lessons)
        run_ocr(videos, tag=tag, use_gpu=not args.cpu)
    elif args.part is not None:
        videos, tag = get_videos_by_part(args.part, args.total)
        run_ocr(videos, tag=tag, use_gpu=not args.cpu)
    else:
        show_lesson_summary()
