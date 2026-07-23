"""
indexer.py — Task A3: CLIP Embedding + FAISS Indexing
=======================================================
Đầu vào  : Keyframes (*.jpg) trong data/processed/keyframes/
            + metadata.json  (dict theo tên file: {"video0_frame_1.jpg": {"video_id":..., "timestamp":...}})
Đầu ra   : data/processed/clip_features.npy   — ma trận vector (N x 512)
            data/processed/frame_index.json    — ánh xạ index → {filename, video_id, timestamp}
            data/processed/index.faiss         — FAISS index sẵn sàng tìm kiếm
"""

import json
import re
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# ============================================================
# 0. ĐƯỜNG DẪN DỰ ÁN (tự động tính từ vị trí file này)
# ============================================================
SRC_DIR = Path(__file__).resolve().parent          # src/indexing/
PROJECT_ROOT = SRC_DIR.parent.parent                # gốc dự án
KEYFRAME_DIR = PROJECT_ROOT / "data" / "processed" / "keyframes"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FEATURES = OUTPUT_DIR / "clip_features.npy"
OUTPUT_MAP = OUTPUT_DIR / "frame_index.json"
OUTPUT_FAISS = OUTPUT_DIR / "index.faiss"

BATCH_SIZE = 8


# ============================================================
# 1. LOAD MODEL CLIP
# ============================================================

def load_clip_model(device: str = None):
    """
    Tải mô hình CLIP (ViT-B/32).
    Ưu tiên GPU nếu có, fallback CPU.
    Trả về: (model, preprocess, device, backend)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[CLIP] Loading CLIP model on device: {device.upper()}")

    try:
        # Thử dùng thư viện openai/CLIP (cài từ GitHub)
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("[CLIP] OK - Using openai/CLIP (ViT-B/32)")
        return model, preprocess, device, "openai"
    except ImportError:
        # Fallback: dùng HuggingFace transformers
        print("[CLIP] openai/CLIP not found. Falling back to HuggingFace...")
        from transformers import CLIPProcessor, CLIPModel
        hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        hf_model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return hf_model, processor, device, "huggingface"


# ============================================================
# 2. TRÍCH XUẤT EMBEDDING THEO BATCH (true batching)
# ============================================================

def extract_batch_openai(image_paths, model, preprocess, device):
    """
    Encode CẢ BATCH trong 1 forward pass duy nhất (nhanh hơn nhiều so với
    encode từng ảnh một, đặc biệt quan trọng khi data thật lên hàng chục nghìn ảnh).
    Trả về: (feature_matrix (B,512) float32 đã L2-normalize, list ảnh hợp lệ đã encode)
    """
    images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            img = preprocess(Image.open(img_path).convert("RGB"))
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  [WARN] Error loading {img_path.name}: {e} - skipping.")

    if not images:
        return np.empty((0, 512), dtype=np.float32), []

    batch_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(batch_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

    return features.cpu().numpy().astype(np.float32), valid_paths


def extract_batch_huggingface(image_paths, model, processor, device):
    """Tương đương extract_batch_openai nhưng dùng backend HuggingFace."""
    images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  [WARN] Error loading {img_path.name}: {e} - skipping.")

    if not images:
        return np.empty((0, 512), dtype=np.float32), []

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

    return features.cpu().numpy().astype(np.float32), valid_paths


# ============================================================
# 3. METADATA
# ============================================================

def load_metadata_dict():
    """
    metadata.json thật là dict theo tên file:
    {"video0_frame_1.jpg": {"video_id": "video0", "timestamp": 1.0}, ...}
    """
    if not METADATA_PATH.exists():
        print(f"  [WARN] Không tìm thấy {METADATA_PATH} - frame_index.json sẽ thiếu video_id/timestamp.")
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sort_key(path: Path):
    """
    Sort theo số thứ tự thật trong tên file (video0_frame_2.jpg trước video0_frame_10.jpg),
    thay vì sort theo string (chỉ ảnh hưởng thứ tự đọc/log, không ảnh hưởng tính đúng của FAISS).
    """
    match = re.search(r"(\d+)(?=\.\w+$)", path.stem)
    return (path.parent.name, int(match.group(1)) if match else 0, path.name)


# ============================================================
# 4. PIPELINE CHÍNH: BUILD INDEX
# ============================================================

def build_index(batch_size: int = BATCH_SIZE):
    """
    Pipeline đầy đủ:
      1. Load keyframes từ thư mục
      2. Trích xuất CLIP embedding theo batch thật (1 forward pass / batch)
      3. Ghép metadata.json (video_id, timestamp) vào frame_index.json
      4. Lưu clip_features.npy và frame_index.json
      5. Tạo và lưu FAISS index
    """

    # --- 4.1 Load model ---
    model, preprocessor, device, backend = load_clip_model()

    # --- 4.2 Thu thập danh sách ảnh ---
    frame_paths = sorted(KEYFRAME_DIR.glob("*.jpg"), key=_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh JPG trong: {KEYFRAME_DIR}")
    print(f"\n[INDEX] Found {len(frame_paths)} keyframes.")

    # --- 4.3 Load metadata để join ---
    metadata_dict = load_metadata_dict()

    # --- 4.4 Trích xuất embedding theo batch thật ---
    all_features = []
    frame_names = []  # tên file theo đúng thứ tự đã encode thành công

    print(f"[INDEX] Extracting embeddings (batch_size={batch_size}, backend={backend})...")
    for i in tqdm(range(0, len(frame_paths), batch_size), desc="Embedding"):
        batch = frame_paths[i: i + batch_size]

        if backend == "openai":
            feats, valid_paths = extract_batch_openai(batch, model, preprocessor, device)
        else:
            feats, valid_paths = extract_batch_huggingface(batch, model, preprocessor, device)

        if len(valid_paths) == 0:
            continue

        all_features.append(feats)
        frame_names.extend([p.name for p in valid_paths])

    if not all_features:
        raise RuntimeError("No embeddings could be extracted!")

    # Ghép toàn bộ thành ma trận (N, 512)
    feature_matrix = np.vstack(all_features).astype(np.float32)
    print(f"[INDEX] Embedding matrix: {feature_matrix.shape} (N frames x D dims)")

    # --- 4.5 Lưu clip_features.npy ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_FEATURES), feature_matrix)
    print(f"[SAVE] OK - clip_features.npy -> {OUTPUT_FEATURES}")

    # --- 4.6 Lưu bản đồ index → {filename, video_id, timestamp} ---
    # frame_index.json: { "0": {"filename": "...", "video_id": "...", "timestamp": ...}, ... }
    frame_index_map = {}
    missing_meta = 0
    for i, name in enumerate(frame_names):
        meta = metadata_dict.get(name)
        if meta is None:
            missing_meta += 1
        frame_index_map[str(i)] = {
            "filename": name,
            "video_id": meta.get("video_id") if meta else None,
            "timestamp": meta.get("timestamp") if meta else None,
        }

    if missing_meta:
        print(f"  [WARN] {missing_meta}/{len(frame_names)} frame(s) không tìm thấy trong metadata.json "
              f"(video_id/timestamp = null) - kiểm tra lại tên file có khớp giữa A2 và keyframes/ không.")

    with open(OUTPUT_MAP, "w", encoding="utf-8") as f:
        json.dump(frame_index_map, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] OK - frame_index.json -> {OUTPUT_MAP}")

    # --- 4.7 Tạo FAISS index ---
    dim = feature_matrix.shape[1]  # số chiều vector (512 cho ViT-B/32)

    # Dùng IndexFlatIP (Inner Product) phù hợp với vector đã normalize L2
    # (cosine similarity ≡ inner product khi đã normalize)
    index = faiss.IndexFlatIP(dim)
    index.add(feature_matrix)
    print(f"[FAISS] Added {index.ntotal} vectors to index (dim={dim})")

    # --- 4.8 Lưu FAISS index ---
    faiss.write_index(index, str(OUTPUT_FAISS))
    print(f"[SAVE] OK - index.faiss -> {OUTPUT_FAISS}")

    print("\n" + "=" * 50)
    print("A3 DONE!")
    print(f"   Frames indexed : {len(frame_names)}")
    print(f"   Vector dims    : {dim}")
    print(f"   CLIP backend   : {backend}")
    print(f"   Device         : {device.upper()}")
    print("=" * 50)

    return feature_matrix, frame_index_map, index


# ============================================================
# 5. HÀM TIỆN ÍCH: LOAD LẠI INDEX (dùng cho retrieval sau này)
# ============================================================

def load_index():
    """
    Load lại FAISS index và mapping đã lưu từ đĩa.
    Trả về: (index, frame_index_map, feature_matrix)
    """
    if not OUTPUT_FAISS.exists():
        raise FileNotFoundError("index.faiss not found. Please run build_index() first.")

    index = faiss.read_index(str(OUTPUT_FAISS))
    with open(OUTPUT_MAP, "r", encoding="utf-8") as f:
        frame_index_map = json.load(f)
    feature_matrix = np.load(str(OUTPUT_FEATURES))

    print(f"[LOAD] OK - Index loaded: {index.ntotal} vectors, dim={index.d}")
    return index, frame_index_map, feature_matrix


# ============================================================
# 6. CHẠY TRỰC TIẾP
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  A3 - CLIP Embedding + FAISS Indexer")
    print("=" * 50)
    print(f"  Keyframes : {KEYFRAME_DIR}")
    print(f"  Metadata  : {METADATA_PATH}")
    print(f"  Output    : {OUTPUT_DIR}")
    print()

    build_index(batch_size=BATCH_SIZE)