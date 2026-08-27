"""
run_ocr_worker.py — Chạy OCR phân tán cho nhóm
================================================
Cách dùng:
  Người 1 (phần 1/3):  python -m src.indexing.run_ocr_worker --part 1 --total 3
  Người 2 (phần 2/3):  python -m src.indexing.run_ocr_worker --part 2 --total 3
  Người 3 (phần 3/3):  python -m src.indexing.run_ocr_worker --part 3 --total 3

  Sau khi tất cả xong:  python -m src.indexing.run_ocr_worker --merge

Output mỗi người: data/processed/ocr_part_<part>_of_<total>.json
Output merge:     data/processed/ocr_index.json

Ghi chú:
  - Mặc định dùng EasyOCR (hỗ trợ tiếng Việt tốt hơn Tesseract)
  - Nếu không có GPU, thêm --cpu để ép dùng CPU (chậm hơn ~5x)
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
    """Lấy toàn bộ thư mục video, sort để đảm bảo tất cả máy chia giống nhau."""
    dirs = sorted([d for d in BTC_KEYFRAMES.iterdir() if d.is_dir()])
    return dirs


def get_my_videos(part: int, total: int) -> list:
    """Trả về danh sách video folder của phần này."""
    all_videos = get_all_video_dirs()
    n = len(all_videos)
    chunk_size = (n + total - 1) // total  # ceiling division
    start = (part - 1) * chunk_size
    end = min(start + chunk_size, n)
    assigned = all_videos[start:end]
    print(f"[OCR Worker] Phan {part}/{total}: video #{start+1} -> #{end} "
          f"({len(assigned)} video, tong {n} video)")
    return assigned


# ──────────────────────────────────────────────
# 2. LOAD OCR ENGINE
# ──────────────────────────────────────────────

def load_easyocr(use_gpu: bool = True):
    """Load EasyOCR reader. Ho tro tieng Viet + tieng Anh."""
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
    """Chay EasyOCR tren 1 anh, tra ve chuoi text da ghep."""
    try:
        results = reader.readtext(str(img_path), detail=0, paragraph=True)
        text = " ".join(results).strip()
        return text
    except Exception:
        return ""


# ──────────────────────────────────────────────
# 4. PIPELINE CHÍNH
# ──────────────────────────────────────────────

def run_ocr_for_part(part: int, total: int, use_gpu: bool = True):
    """
    Chay OCR cho phan duoc phan cong.
    - Resume tu dong: video da xong se bi skip.
    - Luu tung video ngay sau khi xong (khong mat du lieu neu bi ngat).
    """
    from tqdm import tqdm

    output_file = PROCESSED_DIR / f"ocr_part_{part}_of_{total}.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load file da co (de resume)
    ocr_data: dict = {}
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            ocr_data = json.load(f)
        print(f"[Resume] Da co {len(ocr_data)} video trong {output_file.name}")

    # Load EasyOCR
    reader = load_easyocr(use_gpu=use_gpu)

    # Danh sach video can xu ly
    my_videos = get_my_videos(part, total)
    pending = [v for v in my_videos if v.name not in ocr_data]
    print(f"[OCR Worker] Can xu ly: {len(pending)}/{len(my_videos)} video "
          f"(da skip {len(my_videos) - len(pending)} video da co)")

    total_imgs = sum(len(list(v.glob("*.jpg"))) for v in pending)
    print(f"[OCR Worker] Tong anh can OCR: {total_imgs:,}")

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

        # Luu ngay sau khi xong tung video (bao ve khoi crash)
        ocr_data[video_id] = video_ocr
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    has_text = sum(len(v) for v in ocr_data.values())

    print(f"\n{'='*55}")
    print(f"  OCR WORKER PART {part}/{total} - XONG!")
    print(f"{'='*55}")
    print(f"  Video xu ly  : {len(my_videos)}")
    print(f"  Anh da OCR   : {processed_imgs:,}")
    print(f"  Anh co text  : {has_text:,} ({has_text/max(processed_imgs,1)*100:.1f}%)")
    print(f"  Thoi gian     : {elapsed/60:.1f} phut")
    print(f"  Output file   : {output_file}")
    print(f"{'='*55}")
    print(f"\nGui file '{output_file.name}' cho nguoi merge,")
    print(f"sau do chay: python -m src.indexing.run_ocr_worker --merge")


# ──────────────────────────────────────────────
# 5. MERGE TẤT CẢ FILE PARTIAL
# ──────────────────────────────────────────────

def merge_all_parts():
    """
    Gom toan bo file ocr_part_*.json thanh 1 file ocr_index.json.
    Chay sau khi tat ca thanh vien da xong va copy file vao data/processed/.
    """
    part_files = sorted(PROCESSED_DIR.glob("ocr_part_*.json"))
    if not part_files:
        print("[Merge] Khong tim thay file ocr_part_*.json trong data/processed/")
        print("Huong dan: Moi thanh vien copy file ocr_part_X_of_Y.json vao:")
        print(f"  {PROCESSED_DIR}")
        return

    print(f"[Merge] Tim thay {len(part_files)} file partial:")
    for f in part_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")

    merged: dict = {}
    total_videos = 0
    total_frames = 0

    for part_file in part_files:
        with open(part_file, encoding="utf-8") as f:
            data = json.load(f)
        for video_id, frames in data.items():
            merged[video_id] = frames
            total_videos += 1
            total_frames += len(frames)
        n_frames = sum(len(v) for v in data.values())
        print(f"  OK {part_file.name}: {len(data)} video, {n_frames} frame co OCR text")

    with open(OCR_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    size_mb = OCR_INDEX_FILE.stat().st_size / 1024 / 1024
    print(f"\n{'='*55}")
    print(f"  MERGE XONG!")
    print(f"{'='*55}")
    print(f"  Tong video   : {total_videos}")
    print(f"  Frame co text: {total_frames:,}")
    print(f"  File output  : {OCR_INDEX_FILE}")
    print(f"  Kich thuoc   : {size_mb:.1f} MB")
    print(f"{'='*55}")


# ──────────────────────────────────────────────
# 6. KẾ HOẠCH CHIA
# ──────────────────────────────────────────────

def show_split_plan(total: int):
    """In ra ke hoach chia cho N nguoi de tham khao truoc khi chay."""
    all_videos = get_all_video_dirs()
    n = len(all_videos)
    print(f"\n{'='*60}")
    print(f"  KE HOACH CHIA OCR: {n} video --> {total} nguoi")
    print(f"{'='*60}")
    for part in range(1, total + 1):
        chunk_size = (n + total - 1) // total
        start = (part - 1) * chunk_size
        end = min(start + chunk_size, n)
        count = end - start
        sample_dir = all_videos[start] if start < n else None
        sample_imgs = len(list(sample_dir.glob("*.jpg"))) if sample_dir else 180
        est_imgs = count * sample_imgs
        est_gpu = est_imgs * 0.4 / 3600   # ~0.4s/img GPU
        est_cpu = est_imgs * 1.5 / 3600   # ~1.5s/img CPU
        print(f"  Nguoi {part}: video {start+1:4d}~{end:4d} "
              f"({count:3d} video, ~{est_imgs:,} anh, "
              f"GPU~{est_gpu:.0f}h / CPU~{est_cpu:.0f}h)")
    print(f"{'='*60}")
    print(f"\nLenh chay cho tung nguoi:")
    for part in range(1, total + 1):
        print(f"  Nguoi {part}: python -m src.indexing.run_ocr_worker "
              f"--part {part} --total {total}")
    print(f"\nSau khi xong het: python -m src.indexing.run_ocr_worker --merge")


# ──────────────────────────────────────────────
# 7. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR Worker - Chay phan tan cho nhom AIC 2026",
    )
    parser.add_argument("--part", type=int, default=1,
                        help="So thu tu phan cua ban (bat dau tu 1)")
    parser.add_argument("--total", type=int, default=3,
                        help="Tong so nguoi trong nhom")
    parser.add_argument("--merge", action="store_true",
                        help="Merge tat ca file partial thanh ocr_index.json")
    parser.add_argument("--plan", action="store_true",
                        help="Xem ke hoach chia truoc khi chay")
    parser.add_argument("--cpu", action="store_true",
                        help="Ep dung CPU thay vi GPU (cham hon ~5x)")
    args = parser.parse_args()

    if args.merge:
        merge_all_parts()
    elif args.plan:
        show_split_plan(args.total)
    else:
        if args.part < 1 or args.part > args.total:
            parser.error(f"--part phai tu 1 den {args.total}")
        run_ocr_for_part(
            part=args.part,
            total=args.total,
            use_gpu=not args.cpu,
        )
