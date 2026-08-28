"""
run_ocr_worker.py — Chạy OCR phân tán cho nhóm
================================================
Cách dùng:
  1. Chạy theo Lesson (Khuyên dùng):
     Người 1:  python -m src.indexing.run_ocr_worker --lessons L21,L22,L23,L24
     Người 2:  python -m src.indexing.run_ocr_worker --lessons L25
     Người 3:  python -m src.indexing.run_ocr_worker --lessons L26

  2. Chạy theo phân chia tự động (Part X of Total):
     python -m src.indexing.run_ocr_worker --part 1 --total 3

  3. Sau khi tất cả chạy xong:
     python -m src.indexing.run_ocr_worker --merge

Output mỗi người: data/processed/ocr_part_<tag>.json
Output merge:     data/processed/ocr_index.json

Ghi chú:
  - Script tự resume: nếu bị ngắt giữa chừng, chạy lại sẽ bỏ qua video đã xong
  - Cài EasyOCR: pip install easyocr
"""

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BTC_KEYFRAMES = PROJECT_ROOT / "data" / "btc" / "Keyframes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OCR_INDEX_FILE = PROCESSED_DIR / "ocr_index.json"


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
    chunk_size = (n + total - 1) // total  # ceiling division
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
# 4. PIPELINE CHÍNH
# ──────────────────────────────────────────────

def run_ocr(target_videos: list, tag: str, use_gpu: bool = True):
    """
    Chạy OCR cho danh sách video được phân công.
    - Resume tự động: video đã xong trong file JSON sẽ bị skip.
    - Lưu từng video ngay sau khi xong (không mất dữ liệu nếu bị ngắt).
    """
    from tqdm import tqdm

    if not target_videos:
        print("[OCR] Khong tim thay video nao phu hop de chay OCR!")
        return

    output_file = PROCESSED_DIR / f"ocr_part_{tag}.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load file đã có (để resume)
    ocr_data: dict = {}
    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                ocr_data = json.load(f)
            print(f"[Resume] Da co {len(ocr_data)} video trong {output_file.name}")
        except Exception:
            ocr_data = {}

    # Load EasyOCR
    reader = load_easyocr(use_gpu=use_gpu)

    # Danh sách video cần xử lý
    pending = [v for v in target_videos if v.name not in ocr_data]
    print(f"[OCR Worker] Can xu ly: {len(pending)}/{len(target_videos)} video "
          f"(da skip {len(target_videos) - len(pending)} video da co)")

    total_imgs = sum(len(list(v.glob("*.jpg"))) for v in pending)
    print(f"[OCR Worker] Tong so anh can OCR: {total_imgs:,}")

    t0 = time.time()
    processed_imgs = 0

    for v_dir in tqdm(pending, desc="Video", unit="video"):
        video_id = v_dir.name
        img_files = sorted(v_dir.glob("*.jpg"))

        video_ocr: dict = {}
        for img_path in tqdm(img_files, desc=video_id, unit="img", leave=False):
            text = ocr_image(reader, img_path)
            if text:
                video_ocr[img_path.name] = text
            processed_imgs += 1

        # Lưu ngay sau khi xong từng video (bảo vệ khỏi crash/timeout)
        ocr_data[video_id] = video_ocr
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    has_text = sum(len(v) for v in ocr_data.values())

    print(f"\n{'='*55}")
    print(f"  OCR WORKER [{tag}] - XONG!")
    print(f"{'='*55}")
    print(f"  Video xu ly  : {len(target_videos)}")
    print(f"  Anh da OCR   : {processed_imgs:,}")
    print(f"  Anh co text  : {has_text:,} ({has_text/max(processed_imgs,1)*100:.1f}%)")
    print(f"  Thoi gian    : {elapsed/60:.1f} phut")
    print(f"  Output file  : {output_file}")
    print(f"{'='*55}")
    print(f"\nBuoc tiep theo: Tai file '{output_file.name}' ve may de merge:")
    print(f"  python -m src.indexing.run_ocr_worker --merge")


# ──────────────────────────────────────────────
# 5. MERGE TẤT CẢ FILE PARTIAL
# ──────────────────────────────────────────────

def merge_all_parts():
    """
    Gom toàn bộ file ocr_part_*.json thành 1 file ocr_index.json.
    Chạy sau khi tất cả thành viên đã xong và copy file vào data/processed/.
    """
    part_files = sorted(PROCESSED_DIR.glob("ocr_part_*.json"))
    if not part_files:
        print("[Merge] Khong tim thay file ocr_part_*.json trong data/processed/")
        print("Huong dan: Copy cac file ocr_part_*.json vao:")
        print(f"  {PROCESSED_DIR}")
        return

    print(f"[Merge] Tim thay {len(part_files)} file partial:")
    for f in part_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")

    merged: dict = {}
    total_videos = 0
    total_frames = 0

    for part_file in part_files:
        try:
            with open(part_file, encoding="utf-8") as f:
                data = json.load(f)
            for video_id, frames in data.items():
                merged[video_id] = frames
                total_videos += 1
                total_frames += len(frames)
            n_frames = sum(len(v) for v in data.values())
            print(f"  [OK] Da nap {part_file.name}: {len(data)} video, {n_frames} frame co OCR")
        except Exception as e:
            print(f"  [WARN] Loi doc file {part_file.name}: {e}")

    with open(OCR_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

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
        description="OCR Worker - Chay OCR theo Lesson hoac phan chia tu dong",
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
        # Mặc định in kế hoạch
        show_lesson_summary()
