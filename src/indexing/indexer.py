"""
indexer.py — Task A3 + A5: CLIP Embedding + FAISS Indexing
===========================================================
Tich hop A5 (preprocess.py + group_scenes.py):
  - Mac dinh dung keyframes_clean/ + metadata_clean.json (output cua A5)
  - Tuy chon chi index frame dai dien (is_representative=True) de giam FAISS
  - load_index(source=) switch giong load_dataset() trong preprocess.py

Dau vao  : data/processed/keyframes_clean/*.jpg  (output A5)
            data/processed/metadata_clean.json
Dau ra   : data/processed/clip_features_clean.npy
            data/processed/frame_index_clean.json
            data/processed/index_clean.faiss
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
# 0. DUONG DAN DU AN (tu dong tinh tu vi tri file nay)
# ============================================================
SRC_DIR      = Path(__file__).resolve().parent   # src/indexing/
PROJECT_ROOT = SRC_DIR.parent.parent             # goc du an
DATA_DIR     = PROJECT_ROOT / "data" / "processed"

# --- Cac nguon dataset ---
# 'clean' : output A5 preprocess (flat dir, co filter)
# 'sample': output A2 ingest cu (flat dir, chua filter)
# 'self'  : output ingest.py moi (subfolder per video, format BTC)
KEYFRAME_DIR  = DATA_DIR / "keyframes_clean"  # default: clean
METADATA_PATH = DATA_DIR / "metadata_clean.json"
OUTPUT_DIR    = DATA_DIR

OUTPUT_FEATURES = OUTPUT_DIR / "clip_features_clean.npy"
OUTPUT_MAP      = OUTPUT_DIR / "frame_index_clean.json"
OUTPUT_FAISS    = OUTPUT_DIR / "index_clean.faiss"

BATCH_SIZE = 16


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

    🚀 Tối ưu: Dùng torch.cuda.amp.autocast() (fp16) trên GPU → nhanh hơn ~1.7x,
    không ảnh hưởng độ chính xác sau khi L2-normalize.
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

    # fp16 autocast trên GPU → nhanh hơn ~1.7x, fallback float32 trên CPU
    use_amp = device != "cpu" and torch.cuda.is_available()
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                features = model.encode_image(batch_tensor)
        else:
            features = model.encode_image(batch_tensor)
        features = features.float()  # đưa về float32 trước normalize
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

def load_metadata_dict(metadata_path: Path = METADATA_PATH) -> dict:
    """
    metadata_clean.json (A5) la dict theo ten file:
    {
      "video0_frame_1.jpg": {
        "video_id": "video0", "timestamp": 1.0,
        "blur_score": 120.5, "is_clean": true,
        "scene_id": 3, "is_representative": true
      }, ...
    }
    Tuong thich nguoc voi metadata.json cu (A2) — chi can co video_id + timestamp.
    """
    if not metadata_path.exists():
        print(f"  [WARN] Khong tim thay {metadata_path} — frame_index.json se thieu video_id/timestamp.")
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sort_key(path: Path):
    """
    Sort theo so thu tu that trong ten file (video0_frame_2.jpg truoc video0_frame_10.jpg),
    thay vi sort theo string (chi anh huong thu tu doc/log, khong anh huong tinh dung cua FAISS).
    """
    match = re.search(r"(\d+)(?=\.\w+$)", path.stem)
    return (path.parent.name, int(match.group(1)) if match else 0, path.name)


def _get_frame_paths(keyframe_dir: Path, metadata_path: Path,
                     rep_only: bool = False) -> list:
    """
    Lay danh sach frame can index.
    Ho tro 2 cau truc:
      - Flat  : keyframes/*.jpg (output A5 cu)
      - Subfolder: keyframes/<video_id>/*.jpg (output ingest.py moi, format BTC)
    """
    if rep_only:
        try:
            from src.indexing.group_scenes import get_representative_paths
            paths = get_representative_paths(
                metadata_path=metadata_path,
                clean_dir=keyframe_dir,
            )
            if paths:
                return sorted(paths, key=_sort_key)
            print("  [WARN] rep_only=True nhung khong tim thay frame dai dien."
                  " Fallback: dung toan bo frame.")
        except Exception as e:
            print(f"  [WARN] Khong the load group_scenes: {e}. Fallback: toan bo frame.")

    # Flat format: keyframes/*.jpg
    flat_files = list(keyframe_dir.glob("*.jpg"))
    if flat_files:
        return sorted(flat_files, key=_sort_key)

    # Subfolder format: keyframes/<video_id>/*.jpg (BTC / ingest.py moi)
    sub_files = list(keyframe_dir.glob("*/*.jpg"))
    if sub_files:
        print(f"  [INFO] Dung subfolder format ({keyframe_dir.name}/<video_id>/*.jpg)")
        return sorted(sub_files, key=lambda p: (p.parent.name, p.stem))

    return []


# ============================================================
# 4. PIPELINE CHÍNH: BUILD INDEX
# ============================================================

def build_index(
    batch_size: int = BATCH_SIZE,
    keyframe_dir: Path = KEYFRAME_DIR,
    metadata_path: Path = METADATA_PATH,
    output_features: Path = OUTPUT_FEATURES,
    output_map: Path = OUTPUT_MAP,
    output_faiss: Path = OUTPUT_FAISS,
    rep_only: bool = False,
    per_video: bool = False,
):
    """
    Pipeline day du encode CLIP + luu output.

    per_video=False (default):
      Luu 1 file clip_features.npy duy nhat chua tat ca video.
      Dung khi chi 1 may xu ly tat ca.

    per_video=True:
      Luu moi video thanh 1 file rieng:
        data/processed/clip-features/<video_id>.npy
      Dung khi nhieu may xu ly song song, sau do gop lai bang
        load_btc_data.py --mode build-self
    """

    # --- 4.1 Load model ---
    model, preprocessor, device, backend = load_clip_model()

    # --- 4.2 Thu thap danh sach anh ---
    frame_paths = _get_frame_paths(keyframe_dir, metadata_path, rep_only=rep_only)
    if not frame_paths:
        raise FileNotFoundError(f"Khong tim thay anh JPG trong: {keyframe_dir}")
    mode_label = "(representative only)" if rep_only else "(all clean frames)"
    print(f"\n[INDEX] Found {len(frame_paths)} keyframes {mode_label}")

    # --- 4.3 Load metadata de join ---
    metadata_dict = load_metadata_dict(metadata_path)

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
        for p in valid_paths:
            if p.parent.name not in ("keyframes", "keyframes_clean"):
                frame_names.append(f"{p.parent.name}/{p.name}")
            else:
                frame_names.append(p.name)

    if not all_features:
        raise RuntimeError("No embeddings could be extracted!")

    # Ghép toàn bộ thành ma trận (N, 512)
    feature_matrix = np.vstack(all_features).astype(np.float32)
    print(f"[INDEX] Embedding matrix: {feature_matrix.shape} (N frames x D dims)")

    # --- 4.5 Luu clip_features.npy ---
    output_features.parent.mkdir(parents=True, exist_ok=True)

    if per_video:
        # Luu moi video 1 file rieng → dung cho distributed processing
        clip_feat_dir = DATA_DIR.parent / "processed" / "clip-features"
        clip_feat_dir.mkdir(parents=True, exist_ok=True)
        # Nhom frames theo video_id
        video_groups = {}
        for i, name in enumerate(frame_names):
            meta = metadata_dict.get(name, {})
            vid = meta.get("video_id", Path(name).parent.name if "/" in name else "unknown")
            video_groups.setdefault(vid, []).append(i)
        for vid, idxs in video_groups.items():
            vfeats = feature_matrix[idxs]
            vpath  = clip_feat_dir / f"{vid}.npy"
            np.save(str(vpath), vfeats)
        print(f"[SAVE] OK - Per-video .npy -> {clip_feat_dir} ({len(video_groups)} files)")
    else:
        np.save(str(output_features), feature_matrix)
        print(f"[SAVE] OK - {output_features.name} -> {output_features}")

    # --- 4.6 Luu ban do index -> {filename, video_id, timestamp, scene_id, is_representative} ---
    # frame_index_clean.json: { "0": {"filename":..., "video_id":..., "timestamp":..., "scene_id":..., ...}, ... }
    frame_index_map = {}
    missing_meta = 0
    for i, name in enumerate(frame_names):
        meta = metadata_dict.get(name)
        if meta is None:
            missing_meta += 1
        frame_index_map[str(i)] = {
            "filename":          name,
            "video_id":          meta.get("video_id")          if meta else None,
            "timestamp":         meta.get("timestamp")         if meta else None,
            "scene_id":          meta.get("scene_id")          if meta else None,
            "is_representative": meta.get("is_representative") if meta else None,
            "blur_score":        meta.get("blur_score")        if meta else None,
        }

    if missing_meta:
        print(f"  [WARN] {missing_meta}/{len(frame_names)} frame(s) khong tim thay trong metadata.")

    with open(output_map, "w", encoding="utf-8") as f:
        json.dump(frame_index_map, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] OK - {output_map.name} -> {output_map}")

    # --- 4.7 Tạo FAISS index ---
    dim = feature_matrix.shape[1]  # số chiều vector (512 cho ViT-B/32)

    # Dùng IndexFlatIP (Inner Product) phù hợp với vector đã normalize L2
    # (cosine similarity ≡ inner product khi đã normalize)
    index = faiss.IndexFlatIP(dim)
    index.add(feature_matrix)
    print(f"[FAISS] Added {index.ntotal} vectors to index (dim={dim})")

    # --- 4.8 Luu FAISS index ---
    faiss.write_index(index, str(output_faiss))
    print(f"[SAVE] OK - {output_faiss.name} -> {output_faiss}")

    print("\n" + "=" * 50)
    print("A3+A5 INDEXING DONE!")
    print(f"   Frames indexed : {len(frame_names)}")
    print(f"   Rep-only mode  : {rep_only}")
    print(f"   Vector dims    : {dim}")
    print(f"   CLIP backend   : {backend}")
    print(f"   Device         : {device.upper()}")
    print("=" * 50)

    return feature_matrix, frame_index_map, index


# ============================================================
# 5. HÀM TIỆN ÍCH: LOAD LẠI INDEX (dùng cho retrieval sau này)
# ============================================================

def load_index(source: str = "clean"):
    """
    Load lai FAISS index va mapping da luu tu dia.
    Tich hop voi load_dataset() cua preprocess.py.

    Args:
        source: "clean" (default, output A5) | "sample" (data thu A2)

    Returns:
        (index, frame_index_map, feature_matrix)

    Example:
        index, frame_map, feats = load_index("clean")
        # Sau do dung trong retriever.py
    """
    try:
        from src.indexing.preprocess import load_dataset
        ds = load_dataset(source)
        faiss_path    = ds["faiss_index"]
        map_path      = ds["frame_map"]
        features_path = ds["features"]
    except Exception:
        # Fallback ve duong dan mac dinh
        faiss_path    = OUTPUT_FAISS
        map_path      = OUTPUT_MAP
        features_path = OUTPUT_FEATURES

    if not faiss_path.exists():
        raise FileNotFoundError(
            f"{faiss_path.name} not found. Please run build_index() first."
        )

    index = faiss.read_index(str(faiss_path))
    with open(map_path, "r", encoding="utf-8") as f:
        frame_index_map = json.load(f)
    feature_matrix = np.load(str(features_path))

    print(f"[LOAD] OK - source='{source}' | {index.ntotal} vectors, dim={index.d}")
    return index, frame_index_map, feature_matrix


# ============================================================
# 6. CHẠY TRỰC TIẾP
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A3+A5: CLIP Embedding + FAISS Indexer")
    parser.add_argument(
        "--source", choices=["clean", "sample", "self"], default="self",
        help=(
            "Dataset to index:\n"
            "  'self'  : output cua ingest.py moi (subfolder, BTC format) [default]\n"
            "  'clean' : output A5 (keyframes_clean/)\n"
            "  'sample': output ingest cu (keyframes/ phang)"
        )
    )
    parser.add_argument(
        "--rep-only", action="store_true",
        help="Only index representative frames (is_representative=True from A5 grouping)"
    )
    parser.add_argument(
        "--per-video", action="store_true",
        help=(
            "Luu moi video 1 file .npy rieng vao data/processed/clip-features/.\n"
            "Dung khi nhieu may xu ly song song, sau do gop bang:\n"
            "  python -m src.indexing.load_btc_data --mode build-self"
        )
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size for CLIP encoding (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    # Chon duong dan theo source
    if args.source == "self":
        # Output cua ingest.py moi: subfolder per video
        kf_dir   = DATA_DIR / "keyframes"       # data/processed/keyframes/<video_id>/
        meta     = DATA_DIR / "metadata.json"   # tuong thich nguoc
        out_feat = DATA_DIR / "clip_features_self.npy"
        out_map  = DATA_DIR / "frame_index_self.json"
        out_fais = DATA_DIR / "index_self.faiss"
    elif args.source == "sample":
        kf_dir   = DATA_DIR / "keyframes"
        meta     = DATA_DIR / "metadata.json"
        out_feat = DATA_DIR / "clip_features.npy"
        out_map  = DATA_DIR / "frame_index.json"
        out_fais = DATA_DIR / "index.faiss"
    else:  # clean (default A5)
        kf_dir   = KEYFRAME_DIR
        meta     = METADATA_PATH
        out_feat = OUTPUT_FEATURES
        out_map  = OUTPUT_MAP
        out_fais = OUTPUT_FAISS

    print("=" * 55)
    print("  CLIP Embedding + FAISS Indexer")
    print("=" * 55)
    print(f"  Source    : {args.source}")
    print(f"  Rep-only  : {args.rep_only}")
    print(f"  Keyframes : {kf_dir}")
    print(f"  Output    : {DATA_DIR}")
    print()

    build_index(
        batch_size=args.batch_size,
        keyframe_dir=kf_dir,
        metadata_path=meta,
        output_features=out_feat,
        output_map=out_map,
        output_faiss=out_fais,
        rep_only=args.rep_only,
        per_video=args.per_video,
    )

    print()
    print("BUOC TIEP THEO (sau khi co index):")
    print("  python -m src.indexing.load_btc_data --mode build  (neu dung data BTC)")
    print("  python -m src.retrieval.retriever                  (test retrieval)")