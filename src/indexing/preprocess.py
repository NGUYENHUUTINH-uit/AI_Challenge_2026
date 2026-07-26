"""
preprocess.py — Task A5: Preprocessing Pipeline
================================================
Lọc frame nhiễu (blur, dark, overexposed), chuẩn hóa kích thước
và cung cấp hàm load_dataset() để switch nhanh giữa các tập dữ liệu.

Input : data/processed/keyframes/*.jpg  (output của A2/ingest.py)
Output: data/processed/keyframes_clean/ — folder frame đã lọc & chuẩn hóa
        data/processed/metadata_clean.json — metadata nâng cao
              (thêm trường: blur_score, brightness, is_clean)

Chạy trực tiếp:
    python src/indexing/preprocess.py
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
# 0. ĐƯỜNG DẪN DỰ ÁN
# ══════════════════════════════════════════════════════════════════

SRC_DIR      = Path(__file__).resolve().parent          # src/indexing/
PROJECT_ROOT = SRC_DIR.parent.parent                    # gốc dự án
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

KEYFRAME_DIR = DATA_DIR / "keyframes"
CLEAN_DIR    = DATA_DIR / "keyframes_clean"
METADATA_IN  = DATA_DIR / "metadata.json"
METADATA_OUT = DATA_DIR / "metadata_clean.json"

# ══════════════════════════════════════════════════════════════════
# 1. NGƯỠNG LỌC (có thể tune tùy chất lượng camera)
# ══════════════════════════════════════════════════════════════════

BLUR_THRESHOLD  = 50.0    # Laplacian variance — dưới ngưỡng = nhòe/rung
MIN_BRIGHTNESS  = 20.0    # Độ sáng trung bình tối thiểu (tối đen)
MAX_BRIGHTNESS  = 235.0   # Độ sáng trung bình tối đa (trắng xóa / glare)
TARGET_SIZE     = (720, 404)   # (width, height) chuẩn hóa đầu ra


# ══════════════════════════════════════════════════════════════════
# 2. HÀM ĐO & LỌC ĐƠN
# ══════════════════════════════════════════════════════════════════

def compute_blur_score(img_bgr: np.ndarray) -> float:
    """
    Trả về phương sai Laplacian (Laplacian Variance).
    - Giá trị THẤP  → frame nhòe / rung / chuyển cảnh
    - Giá trị CAO   → frame sắc nét
    Tham khảo: Pech-Pacheco et al. (2000) — chuẩn công nghiệp.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(img_bgr: np.ndarray) -> float:
    """Trả về độ sáng trung bình kênh xám (0–255)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_blurry(img_bgr: np.ndarray,
              threshold: float = BLUR_THRESHOLD) -> bool:
    """True nếu frame bị nhòe (Laplacian variance < threshold)."""
    return compute_blur_score(img_bgr) < threshold


def is_bad_exposure(img_bgr: np.ndarray,
                    min_b: float = MIN_BRIGHTNESS,
                    max_b: float = MAX_BRIGHTNESS) -> bool:
    """True nếu frame quá tối (< min_b) hoặc quá sáng (> max_b)."""
    b = compute_brightness(img_bgr)
    return b < min_b or b > max_b


def normalize_frame(img_bgr: np.ndarray,
                    target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    Chuẩn hóa frame:
      - Resize về target_size (giữ tỷ lệ, pad đen hai bên nếu cần)
      - Giữ nguyên BGR (cv2.imwrite dùng BGR)
    Lý do: CLIP yêu cầu ảnh nhất quán kích thước để batch processing
    hiệu quả; padding giữ tỷ lệ tránh méo hình học làm sai embedding.
    """
    h, w = img_bgr.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    # Tạo canvas đen và đặt ảnh vào giữa (letterbox)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y_off = (th - nh) // 2
    x_off = (tw - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    return canvas


# ══════════════════════════════════════════════════════════════════
# 3. PIPELINE LỌC TOÀN BỘ KEYFRAMES
# ══════════════════════════════════════════════════════════════════

def run_filtering(
    keyframe_dir: Path = KEYFRAME_DIR,
    clean_dir: Path = CLEAN_DIR,
    metadata_in: Path = METADATA_IN,
    metadata_out: Path = METADATA_OUT,
    blur_threshold: float = BLUR_THRESHOLD,
    min_brightness: float = MIN_BRIGHTNESS,
    max_brightness: float = MAX_BRIGHTNESS,
    normalize: bool = True,
) -> dict:
    """
    Quét toàn bộ keyframes/, tính chỉ số chất lượng từng frame,
    lọc frame nhiễu, lưu frame sạch vào keyframes_clean/ và
    ghi metadata_clean.json với các trường bổ sung.

    Trả về: metadata_clean (dict) — dùng ngay hoặc bỏ qua.

    Luồng xử lý mỗi frame:
        load → compute_blur + compute_brightness
             → if clean: normalize → lưu vào clean_dir
             → ghi metadata (kể cả frame bị loại, is_clean=False)
    """
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata gốc từ A2
    if metadata_in.exists():
        with open(metadata_in, "r", encoding="utf-8") as f:
            meta_orig = json.load(f)
        print(f"[A5] Loaded metadata: {len(meta_orig)} entries from {metadata_in.name}")
    else:
        print(f"  [WARN] metadata.json not found at {metadata_in} — sẽ bỏ qua video_id/timestamp")
        meta_orig = {}

    all_frames = sorted(keyframe_dir.glob("*.jpg"))
    if not all_frames:
        raise FileNotFoundError(
            f"Không tìm thấy file .jpg trong: {keyframe_dir}\n"
            f"Hãy chạy ingest.py (A2) trước để trích keyframe."
        )

    print(f"[A5] Found {len(all_frames)} raw keyframes in {keyframe_dir.name}/".encode('ascii', 'replace').decode('ascii'))

    metadata_clean = {}
    stats = {
        "total":          len(all_frames),
        "blurry":         0,
        "bad_brightness": 0,
        "both":           0,
        "kept":           0,
    }

    for img_path in tqdm(all_frames, desc="[A5] Filtering"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name} — skipping.")
            continue

        blur_score  = compute_blur_score(img)
        brightness  = compute_brightness(img)

        blurry    = blur_score < blur_threshold
        bad_light = brightness < min_brightness or brightness > max_brightness
        is_clean  = not blurry and not bad_light

        # Thống kê
        if blurry and bad_light:
            stats["both"] += 1
        elif blurry:
            stats["blurry"] += 1
        elif bad_light:
            stats["bad_brightness"] += 1

        # Lấy thông tin gốc từ A2 (video_id, timestamp)
        orig_meta = meta_orig.get(img_path.name, {})

        metadata_clean[img_path.name] = {
            **orig_meta,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "is_clean":   is_clean,
        }

        if not is_clean:
            continue

        # Lưu frame sạch (chuẩn hóa letterbox nếu cần)
        out_img  = normalize_frame(img) if normalize else img
        out_path = clean_dir / img_path.name
        cv2.imwrite(str(out_path), out_img)
        stats["kept"] += 1

    # Ghi metadata nâng cao
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata_clean, f, ensure_ascii=False, indent=2)

    _print_stats(stats)
    print(f"[A5] Clean frames        → {clean_dir}")
    print(f"[A5] metadata_clean.json → {metadata_out}")
    return metadata_clean


def _print_stats(stats: dict):
    total = stats["total"]
    kept  = stats["kept"]
    pct   = 100 * kept / total if total else 0
    removed = total - kept
    print()
    print("=" * 50)
    print("  A5 FILTERING REPORT")
    print("=" * 50)
    print(f"  Total raw frames   : {total}")
    print(f"  Blurry removed     : {stats['blurry']}")
    print(f"  Bad exposure remov.: {stats['bad_brightness']}")
    print(f"  Both issues removed: {stats['both']}")
    print(f"  Total removed      : {removed}  ({100 - pct:.1f}%)")
    print(f"  ✓ Kept (clean)     : {kept}  ({pct:.1f}%)")
    print("=" * 50)
    print()


# ══════════════════════════════════════════════════════════════════
# 4. LOAD SWITCH — chuyển nhanh giữa các dataset
# ══════════════════════════════════════════════════════════════════

# Registry các tập dữ liệu.
# Thêm "real" vào đây khi BTC bàn giao data thật (Giai đoạn 2).
_DATASET_REGISTRY: dict[str, dict] = {
    "sample": {
        "keyframe_dir": DATA_DIR / "keyframes",
        "metadata":     DATA_DIR / "metadata.json",
        "faiss_index":  DATA_DIR / "index.faiss",
        "frame_map":    DATA_DIR / "frame_index.json",
        "features":     DATA_DIR / "clip_features.npy",
    },
    "clean": {
        "keyframe_dir": DATA_DIR / "keyframes_clean",
        "metadata":     DATA_DIR / "metadata_clean.json",
        "faiss_index":  DATA_DIR / "index_clean.faiss",
        "frame_map":    DATA_DIR / "frame_index_clean.json",
        "features":     DATA_DIR / "clip_features_clean.npy",
    },
    # "real": { ... }   ← thêm sau khi có data thật
}


def load_dataset(source: str = "clean", path: str = None) -> dict:
    """
    Trả về dict chứa đường dẫn tất cả thành phần của dataset.
    Cho phép indexer.py và retriever.py switch dataset không cần
    sửa hard-coded path.

    Args:
        source : "sample" | "clean" | "custom"
        path   : chỉ dùng khi source="custom" — thư mục gốc data

    Returns:
        {
          "keyframe_dir" : Path  — thư mục chứa ảnh keyframe,
          "metadata"     : Path  — file metadata JSON,
          "faiss_index"  : Path  — file index.faiss,
          "frame_map"    : Path  — frame_index.json (index → filename),
          "features"     : Path  — clip_features.npy,
        }

    Example:
        from src.indexing.preprocess import load_dataset
        import faiss, json, numpy as np

        ds    = load_dataset("clean")
        index = faiss.read_index(str(ds["faiss_index"]))
        meta  = json.load(open(ds["metadata"]))
        feats = np.load(ds["features"])
    """
    if source == "custom":
        if path is None:
            raise ValueError("source='custom' yêu cầu tham số path='/đường/dẫn'")
        base = Path(path)
        return {
            "keyframe_dir": base / "keyframes_clean",
            "metadata":     base / "metadata_clean.json",
            "faiss_index":  base / "index_clean.faiss",
            "frame_map":    base / "frame_index_clean.json",
            "features":     base / "clip_features_clean.npy",
        }

    if source not in _DATASET_REGISTRY:
        raise ValueError(
            f"source='{source}' không hợp lệ. "
            f"Chọn một trong: {list(_DATASET_REGISTRY.keys())} hoặc 'custom'."
        )

    paths = _DATASET_REGISTRY[source]
    print(f"\n[load_dataset] source='{source}'")
    for key, val in paths.items():
        status = "✓" if val.exists() else "✗  (chưa có — cần chạy pipeline trước)"
        print(f"   {key:<15s}: {val.name}  {status}")
    print()
    return paths


# ══════════════════════════════════════════════════════════════════
# 5. CHẠY TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  A5 — Preprocessing Pipeline (Filtering + Normalize)")
    print("=" * 50)
    print(f"  Input  : {KEYFRAME_DIR}")
    print(f"  Output : {CLEAN_DIR}")
    print(f"  Blur threshold      : {BLUR_THRESHOLD}")
    print(f"  Brightness range    : [{MIN_BRIGHTNESS}, {MAX_BRIGHTNESS}]")
    print(f"  Target size         : {TARGET_SIZE}")
    print()

    run_filtering()
