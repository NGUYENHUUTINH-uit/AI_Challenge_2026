"""
group_scenes.py — Phần nâng cao của A5: Scene Grouping bằng pHash
==================================================================
Gom các keyframe giống nhau (cùng cảnh tĩnh) thành 1 scene_id,
rồi chọn 1 frame đại diện (blur_score cao nhất) cho mỗi scene.

Mục tiêu:
  - Giảm 60–80% số vector trong FAISS với camera surveillance đứng yên
  - Tránh kết quả tìm kiếm bị "đóng cụm" vào cùng một cảnh trùng lặp
  - Bổ sung scene_id vào metadata → Agent dùng để temporal localization

Lý thuyết (pHash — Perceptual Hash):
  1. Resize ảnh → 32×32, grayscale
  2. Áp DCT (Discrete Cosine Transform)
  3. Lấy 64 bit tần số thấp → hash 64-bit
  4. Hamming distance ≤ PHASH_THRESHOLD → cùng scene
  Ưu điểm: nhanh ~100x so với so sánh CLIP embedding, không cần GPU.

Input : data/processed/keyframes_clean/ + metadata_clean.json
        (chạy preprocess.py TRƯỚC)
Output: data/processed/metadata_clean.json (CẬP NHẬT thêm scene_id,
        is_representative)

Chạy trực tiếp:
    python src/indexing/group_scenes.py
"""

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
# 0. ĐƯỜNG DẪN DỰ ÁN
# ══════════════════════════════════════════════════════════════════

SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

CLEAN_DIR    = DATA_DIR / "keyframes_clean"
METADATA_OUT = DATA_DIR / "metadata_clean.json"

# Hamming distance ≤ ngưỡng này → 2 frame được coi là "cùng scene"
# 0  = y hệt nhau (pixel-perfect)
# 10 = gần như y hệt, chấp nhận ánh sáng thay đổi nhẹ
# 20 = cho phép thay đổi vừa (góc quay hơi lệch)
# 64 = hoàn toàn khác nhau
PHASH_THRESHOLD = 10


# ══════════════════════════════════════════════════════════════════
# 1. TÍNH PHASH
# ══════════════════════════════════════════════════════════════════

def compute_phash(img_path: Path):
    """
    Tính Perceptual Hash (pHash) cho 1 ảnh.
    Trả về: imagehash.ImageHash object (có thể dùng toán tử - để tính Hamming distance).
    """
    try:
        import imagehash
    except ImportError:
        raise ImportError(
            "Thiếu thư viện 'imagehash'. Cài bằng:\n"
            "    pip install imagehash"
        )
    return imagehash.phash(Image.open(img_path).convert("RGB"))


# ══════════════════════════════════════════════════════════════════
# 2. GROUPING THEO PHASH
# ══════════════════════════════════════════════════════════════════

def group_by_phash(
    clean_dir: Path = CLEAN_DIR,
    metadata_path: Path = METADATA_OUT,
    threshold: int = PHASH_THRESHOLD,
) -> dict[str, int]:
    """
    Gom frame thành scenes bằng pHash (Sequential Grouping).

    Thuật toán:
      - Với mỗi video, sort frame theo timestamp
      - Duyệt tuần tự: so sánh pHash frame hiện tại vs frame ĐẦU của scene hiện tại
      - Nếu Hamming distance > threshold → cắt scene mới
      - Chỉ gom trong cùng video_id (không bao giờ gom frame từ video khác)

    Trả về:
        frame_to_scene: { "filename.jpg": scene_id (int), ... }
        Chỉ bao gồm frame có is_clean=True và tồn tại trong clean_dir.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {metadata_path}.\n"
            f"Hãy chạy preprocess.py (A5 Filtering) trước."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Chỉ xử lý frame sạch và thực sự tồn tại trên đĩa
    clean_frames = {
        name: info
        for name, info in meta.items()
        if info.get("is_clean", False) and (clean_dir / name).exists()
    }

    if not clean_frames:
        raise RuntimeError(
            "Không tìm thấy frame sạch trong metadata_clean.json hoặc "
            f"thư mục {clean_dir} rỗng. Hãy chạy preprocess.py trước."
        )

    print(f"[group_scenes] {len(clean_frames)} clean frames to process...")

    # Nhóm theo video_id
    by_video: dict[str, list[str]] = defaultdict(list)
    for name, info in clean_frames.items():
        vid = info.get("video_id", "unknown")
        by_video[vid].append(name)

    frame_to_scene: dict[str, int] = {}
    global_scene_id = 0

    for vid, frame_names in by_video.items():
        # Sort theo timestamp (đảm bảo đúng thứ tự thời gian)
        frame_names.sort(key=lambda n: meta[n].get("timestamp", 0))

        current_scene_hash = None   # pHash của frame đầu scene hiện tại
        current_scene_id   = None

        for name in tqdm(frame_names, desc=f"  pHash [{vid}]", leave=False):
            img_path = clean_dir / name
            try:
                ph = compute_phash(img_path)
            except Exception as e:
                # Lỗi hash → tạo scene riêng cho frame này
                print(f"  [WARN] pHash failed for {name}: {e} — solo scene")
                frame_to_scene[name] = global_scene_id
                global_scene_id += 1
                current_scene_hash = None
                current_scene_id   = None
                continue

            if current_scene_hash is None:
                # Frame đầu tiên của video → bắt đầu scene mới
                current_scene_id   = global_scene_id
                global_scene_id   += 1
                current_scene_hash = ph
            else:
                hamming_dist = ph - current_scene_hash
                if hamming_dist > threshold:
                    # Cảnh thay đổi đủ lớn → cắt scene mới
                    current_scene_id   = global_scene_id
                    global_scene_id   += 1
                    current_scene_hash = ph
                # (nếu ≤ threshold → giữ nguyên scene, không cập nhật hash)

            frame_to_scene[name] = current_scene_id

    print(f"[group_scenes] {len(frame_to_scene)} frames → {global_scene_id} scenes")
    return frame_to_scene


# ══════════════════════════════════════════════════════════════════
# 3. CHỌN FRAME ĐẠI DIỆN
# ══════════════════════════════════════════════════════════════════

def select_representatives(
    frame_to_scene: dict[str, int],
    metadata_path: Path = METADATA_OUT,
) -> dict[int, str]:
    """
    Với mỗi scene, chọn frame có blur_score CAO NHẤT làm đại diện.
    Frame rõ nét nhất = thông tin ngữ nghĩa phong phú nhất khi CLIP encode.

    Trả về:
        representatives: { scene_id (int): representative_filename (str) }
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scenes: dict[int, list[str]] = defaultdict(list)
    for name, scene_id in frame_to_scene.items():
        scenes[scene_id].append(name)

    representatives: dict[int, str] = {}
    for scene_id, frames in scenes.items():
        best = max(
            frames,
            key=lambda n: meta.get(n, {}).get("blur_score", 0.0)
        )
        representatives[scene_id] = best

    return representatives


# ══════════════════════════════════════════════════════════════════
# 4. PIPELINE ĐẦY ĐỦ: CẬP NHẬT METADATA
# ══════════════════════════════════════════════════════════════════

def update_metadata_with_scenes(
    clean_dir: Path = CLEAN_DIR,
    metadata_path: Path = METADATA_OUT,
    threshold: int = PHASH_THRESHOLD,
) -> None:
    """
    Pipeline đầy đủ của group_scenes:
      1. Tính pHash từng frame sạch
      2. Group thành scenes (sequential, per-video)
      3. Chọn frame đại diện (blur_score cao nhất)
      4. Cập nhật metadata_clean.json thêm 2 trường mới:
           - "scene_id"         : int  — ID scene mà frame này thuộc về
           - "is_representative": bool — True nếu frame được chọn đại diện scene

    Sau bước này, indexer.py có thể chỉ index frame is_representative=True
    để giảm kích thước FAISS index đáng kể.
    """
    print("[group_scenes] Step 1/3 — Computing pHash and grouping scenes...")
    frame_to_scene = group_by_phash(
        clean_dir=clean_dir,
        metadata_path=metadata_path,
        threshold=threshold,
    )

    print("[group_scenes] Step 2/3 — Selecting representative frames...")
    representatives = select_representatives(frame_to_scene, metadata_path)
    rep_files = set(representatives.values())

    print("[group_scenes] Step 3/3 — Updating metadata_clean.json...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for name in meta:
        if name in frame_to_scene:
            meta[name]["scene_id"]          = frame_to_scene[name]
            meta[name]["is_representative"] = (name in rep_files)
        else:
            # Frame bị lọc (is_clean=False) hoặc không tồn tại
            meta[name]["scene_id"]          = None
            meta[name]["is_representative"] = False

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Thống kê
    total_scenes = len(representatives)
    total_frames = len(frame_to_scene)
    ratio        = total_frames / max(total_scenes, 1)

    print()
    print("=" * 50)
    print("  A5 SCENE GROUPING REPORT")
    print("=" * 50)
    print(f"  Clean frames processed : {total_frames}")
    print(f"  Scenes detected        : {total_scenes}")
    print(f"  Avg frames per scene   : {ratio:.1f}x")
    print(f"  FAISS size reduction   : ~{100 * (1 - 1/ratio):.0f}%  (nếu chỉ index representative)")
    print(f"  pHash threshold used   : {threshold}/64 bits Hamming")
    print("=" * 50)
    print(f"  ✓ metadata_clean.json updated: {metadata_path}")
    print()


# ══════════════════════════════════════════════════════════════════
# 5. HELPER: LỌC CHỈ FRAME ĐẠI DIỆN (dùng trong indexer.py)
# ══════════════════════════════════════════════════════════════════

def get_representative_paths(
    metadata_path: Path = METADATA_OUT,
    clean_dir: Path = CLEAN_DIR,
) -> list[Path]:
    """
    Trả về danh sách Path của các frame đại diện (is_representative=True).
    Dùng trong indexer.py để chỉ embed frame đại diện vào FAISS.

    Example (trong indexer.py):
        from src.indexing.group_scenes import get_representative_paths
        frame_paths = get_representative_paths()
        # thay cho: frame_paths = sorted(KEYFRAME_DIR.glob("*.jpg"))
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    paths = [
        clean_dir / name
        for name, info in meta.items()
        if info.get("is_representative", False)
    ]

    # Lọc chỉ giữ file thực sự tồn tại
    valid_paths = [p for p in paths if p.exists()]
    print(f"[group_scenes] Representative frames: {len(valid_paths)} / {len(paths)} total")
    return valid_paths


# ══════════════════════════════════════════════════════════════════
# 6. CHẠY TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  A5 — Scene Grouping (pHash)")
    print("=" * 50)
    print(f"  Clean frames : {CLEAN_DIR}")
    print(f"  Metadata     : {METADATA_OUT}")
    print(f"  Threshold    : {PHASH_THRESHOLD}/64 bits Hamming")
    print()

    update_metadata_with_scenes()
