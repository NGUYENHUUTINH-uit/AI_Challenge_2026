"""
load_btc_data.py — Loader cho dữ liệu chuẩn BTC AIC 2026
==========================================================
BTC cung cấp sẵn:
  - <VideoID>.npy  : CLIP features shape (N_keyframes, 512) float16, đã L2-normalize
  - <VideoID>.csv  : Frame map, cột: n, pts_time, fps, frame_idx

Cấu trúc thư mục BTC (sau khi tải về):
  data/
  ├── clip-features/
  │   ├── L21_V001.npy
  │   ├── L21_V002.npy
  │   └── ...
  └── map-keyframes/
      ├── L21_V001.csv
      ├── L21_V002.csv
      └── ...

Chạy script này để:
  1. Kiểm tra tính nhất quán giữa .npy và .csv của từng video  (mode=inspect)
  2. Gộp tất cả features thành 1 FAISS index duy nhất           (mode=build)
  3. Test nhanh với 1 cặp file cụ thể                           (mode=test-single)
"""

import csv
import json
import numpy as np
import faiss
from pathlib import Path

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent   # goc du an
DATA_DIR        = PROJECT_ROOT / "data"

# --- Thu muc data BTC (moi nguoi tu tai ve may rieng, khong commit Git) ---
BTC_DIR         = DATA_DIR / "btc"
FEATURES_DIR    = BTC_DIR / "clip-features"    # .npy files tu BTC
MAP_DIR         = BTC_DIR / "map-keyframes"    # .csv files tu BTC
VIDEOS_DIR      = BTC_DIR / "Videos"           # video goc 12GB (chi can xem lai)
KEYFRAMES_DIR   = BTC_DIR / "Keyframes"        # anh keyframe BTC trich san
OBJECTS_DIR     = BTC_DIR / "Objects"          # .json object detection tu Faster R-CNN
MEDIA_INFO_DIR  = BTC_DIR / "media-info"       # .json metadata video (YouTube info, FPS, duration...)

# --- Output cua pipeline (auto-generated, cung gitignored) ---
OUTPUT_DIR      = DATA_DIR / "processed"
OUTPUT_FAISS    = OUTPUT_DIR / "index_btc.faiss"
OUTPUT_MAP      = OUTPUT_DIR / "frame_index_btc.json"
OUTPUT_FEATURES = OUTPUT_DIR / "clip_features_btc.npy"
OUTPUT_OBJ_CACHE = OUTPUT_DIR / "objects_cache_btc.json"   # cache objects da load


# ============================================================
# 0b. LOAD OBJECTS & MEDIA INFO TU BTC
# ============================================================

def load_objects_for_video(video_id: str,
                           objects_dir: Path = OBJECTS_DIR) -> dict:
    """
    Load tat ca object detection cua 1 video tu thu muc Objects cua BTC.
    BTC format: Objects/<video_id>/<frame_name>.json
    Moi file JSON chua list cac detected objects (Faster R-CNN OpenImages V4).

    Returns:
        { "0000": ["person", "car", ...], "0001": [...], ... }
        Key la ten file keyframe (khong co .jpg), value la list ten class.

    Vi du:
        objs = load_objects_for_video("L21_V001")
        print(objs["0001"])  # -> ["person", "bicycle"]
    """
    video_obj_dir = objects_dir / video_id
    if not video_obj_dir.exists():
        return {}

    result = {}
    for json_file in sorted(video_obj_dir.glob("*.json")):
        frame_name = json_file.stem   # "0000", "0001", ...
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            # Format BTC: list cac detected object dicts, moi dict co "class_name"
            # Tuong thich ca format {"objects": [...]} lan format [...] truc tiep
            if isinstance(data, list):
                objs = data
            elif isinstance(data, dict):
                objs = data.get("objects", data.get("detections", []))
            else:
                objs = []
            classes = list({
                obj.get("class", obj.get("class_name", obj.get("label", "")))
                for obj in objs
                if isinstance(obj, dict)
            })
            result[frame_name] = [c for c in classes if c]
        except Exception:
            result[frame_name] = []

    return result


def load_media_info(video_id: str,
                    media_info_dir: Path = MEDIA_INFO_DIR) -> dict:
    """
    Load metadata video (YouTube info, fps, duration, title, ...) tu BTC.
    BTC format: media-info/<video_id>.json

    Returns:
        dict voi cac truong nhu fps, duration, title, description, ...
        Tra ve {} neu khong tim thay file.
    """
    json_path = media_info_dir / f"{video_id}.json"
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================
# 1. KIEM TRA 1 CAP FILE (video_id.npy + video_id.csv)
# ============================================================

def inspect_single_video(npy_path: Path, csv_path: Path, verbose: bool = True) -> dict:
    """
    Kiem tra tinh nhat quan giua 1 file .npy va 1 file .csv cua cung 1 video.
    Returns dict: video_id, n_features, n_keyframes, is_consistent, sample_rows
    """
    video_id = npy_path.stem  # Vi du: "L21_V001"

    # Load features
    features = np.load(str(npy_path))
    n_features = features.shape[0]
    dim = features.shape[1] if features.ndim == 2 else None

    # Kiem tra L2 norm (BTC luon normalize san)
    norms = np.linalg.norm(features.astype(np.float32), axis=1)
    norm_ok = bool(np.allclose(norms, 1.0, atol=0.01))

    # Load CSV
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n_keyframes = len(rows)

    is_consistent = (n_features == n_keyframes)

    if verbose:
        status = "OK" if is_consistent else f"KHONG KHOP ({n_features} features vs {n_keyframes} rows CSV)"
        print(f"[{video_id}] Shape: {features.shape}, dtype: {features.dtype}, "
              f"L2-norm OK: {norm_ok}, CSV rows: {n_keyframes} -> {status}")

    return {
        "video_id":      video_id,
        "n_features":    n_features,
        "n_keyframes":   n_keyframes,
        "dim":           dim,
        "dtype":         str(features.dtype),
        "l2_norm_ok":    norm_ok,
        "is_consistent": is_consistent,
        "sample_rows":   rows[:3],
    }


def inspect_all_videos(features_dir: Path = FEATURES_DIR,
                       map_dir: Path = MAP_DIR) -> list:
    """Quet toan bo thu muc va kiem tra tat ca cap (.npy, .csv)."""
    npy_files = sorted(features_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"Khong tim thay file .npy trong: {features_dir}")

    print(f"\n{'='*60}")
    print(f"  KIEM TRA DU LIEU BTC -- {len(npy_files)} videos")
    print(f"{'='*60}")

    results = []
    n_ok = 0
    n_err = 0
    for npy_path in npy_files:
        csv_path = map_dir / f"{npy_path.stem}.csv"
        if not csv_path.exists():
            print(f"[{npy_path.stem}] WARN: Thieu file CSV tuong ung!")
            n_err += 1
            continue
        info = inspect_single_video(npy_path, csv_path, verbose=True)
        results.append(info)
        if info["is_consistent"]:
            n_ok += 1
        else:
            n_err += 1

    print(f"\nKet qua: {n_ok} OK / {n_err} loi / {len(npy_files)} tong")
    return results


# ============================================================
# 2. XAY FAISS INDEX TU DATA BTC
# ============================================================

def build_btc_index(
    features_dir: Path = FEATURES_DIR,
    map_dir: Path = MAP_DIR,
    output_faiss: Path = OUTPUT_FAISS,
    output_map: Path = OUTPUT_MAP,
    output_features: Path = OUTPUT_FEATURES,
    load_objects: bool = True,       # nhung objects vao frame_index_map
    load_media: bool = True,         # nhung media info vao map
) -> tuple:
    """
    Gop CLIP features cua TAT CA video thanh 1 FAISS index duy nhat.
    Xay frame_index_map:
      {
        "global_idx": {
          video_id, n, pts_time, fps, frame_idx,
          frame_id,      <- gia tri can nop cho BTC (= frame_idx)
          objects,       <- list class from Faster R-CNN (neu load_objects=True)
          duration,      <- tong so giay video (tu media info)
          title,         <- tieu de YouTube (neu co)
        }
      }

    Returns:
        (faiss_index, frame_index_map, feature_matrix)
    """
    npy_files = sorted(features_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"Khong tim thay .npy trong {features_dir}")

    print(f"\n{'='*60}")
    print(f"  BUILD BTC INDEX -- {len(npy_files)} videos")
    print(f"  load_objects={load_objects} | load_media={load_media}")
    print(f"{'='*60}")

    all_features = []
    frame_index_map = {}
    global_idx = 0

    for npy_path in npy_files:
        video_id = npy_path.stem
        csv_path = map_dir / f"{video_id}.csv"

        feats = np.load(str(npy_path)).astype(np.float32)
        n_frames = feats.shape[0]

        # Load CSV
        keyframe_rows = []
        if csv_path.exists():
            with open(csv_path, encoding="utf-8", newline="") as f:
                keyframe_rows = list(csv.DictReader(f))
        else:
            print(f"  [WARN] Khong tim thay {csv_path.name}")
            keyframe_rows = [{}] * n_frames

        if len(keyframe_rows) != n_frames:
            print(f"  [WARN] {video_id}: {n_frames} features != {len(keyframe_rows)} CSV rows -- cat theo min")
            n_frames = min(n_frames, len(keyframe_rows))
            feats = feats[:n_frames]

        # Load objects (Faster R-CNN) -- dict { "0000": ["person",...], ... }
        obj_map = {}
        if load_objects:
            obj_map = load_objects_for_video(video_id)

        # Load media info (YouTube metadata)
        media = {}
        if load_media:
            media = load_media_info(video_id)

        all_features.append(feats)

        for i in range(n_frames):
            row = keyframe_rows[i] if i < len(keyframe_rows) else {}
            # Ten keyframe BTC: so thu tu 4 chu so -> "0000", "0001",...
            # n trong CSV bat dau tu 1, keyframe bat dau tu 0000
            n_val = int(row.get("n", i + 1))
            frame_name = f"{n_val - 1:04d}"   # n=1 -> "0000", n=2 -> "0001"...

            frame_index_map[str(global_idx)] = {
                "video_id":  video_id,
                "n":         n_val,
                "pts_time":  float(row.get("pts_time", 0.0)),
                "fps":       float(row.get("fps", 30.0)),
                "frame_idx": int(row.get("frame_idx", 0)),
                # frame_id = gia tri can nop cho BTC
                "frame_id":  int(row.get("frame_idx", 0)),
                # Objects tu Faster R-CNN (dung cho rerank_by_objects)
                "objects":   obj_map.get(frame_name, []),
                # Media info
                "duration":  media.get("duration", None),
                "title":     media.get("title", ""),
            }
            global_idx += 1

        obj_status = f"{len(obj_map)} frames co objects" if obj_map else "khong co objects"
        print(f"  [{video_id}] {n_frames} frames | {obj_status} | offset: {global_idx - n_frames}")

    feature_matrix = np.vstack(all_features).astype(np.float32)
    print(f"\n[BUILD] Tong feature matrix: {feature_matrix.shape}")

    dim = feature_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(feature_matrix)
    print(f"[FAISS] Da them {index.ntotal} vectors (dim={dim})")

    output_faiss.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_faiss))
    np.save(str(output_features), feature_matrix)
    with open(output_map, "w", encoding="utf-8") as f:
        json.dump(frame_index_map, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {output_faiss.name}")
    print(f"[SAVE] {output_features.name}")
    print(f"[SAVE] {output_map.name}")
    print(f"\n{'='*60}")
    print("  BUILD BTC INDEX DONE!")
    print(f"  Tong vectors : {index.ntotal}")
    print(f"  So video     : {len(npy_files)}")
    print(f"{'='*60}")

    return index, frame_index_map, feature_matrix


# ============================================================
# 3. QUICK TEST VOI 1 VIDEO (dung de debug nhanh + chay Colab)
# ============================================================

def quick_test_single(npy_path: str, csv_path: str):
    """
    Test nhanh voi 1 cap file cu the.
    Co the goi tu Colab hoac local voi duong dan tuy chinh.

    Vi du (Colab):
        from load_btc_data import quick_test_single
        quick_test_single('/content/L21_V001.npy', '/content/L21_V001.csv')
    """
    import csv as csv_mod

    npy_path = Path(npy_path)
    csv_path = Path(csv_path)

    print(f"\n{'='*50}")
    print(f"  QUICK TEST: {npy_path.name} + {csv_path.name}")
    print(f"{'='*50}")

    # --- .npy ---
    features = np.load(str(npy_path)).astype(np.float32)
    print(f"[NPY] Shape   : {features.shape}")
    print(f"[NPY] Dtype   : {features.dtype}")
    print(f"[NPY] Norm[0] : {np.linalg.norm(features[0]):.6f}  (ky vong ~ 1.0)")
    print(f"[NPY] Vec[0][:8]: {features[0][:8]}")

    # --- .csv ---
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv_mod.DictReader(f))
    print(f"\n[CSV] Tong so rows : {len(rows)}")
    print(f"[CSV] Columns      : {list(rows[0].keys()) if rows else 'N/A'}")
    print(f"[CSV] Row 0        : {rows[0]}")
    print(f"[CSV] Row cuoi     : {rows[-1]}")

    # --- Kiem tra nhat quan ---
    n_feats = features.shape[0]
    n_rows  = len(rows)
    match   = "KHOP OK" if n_feats == n_rows else f"KHONG KHOP ({n_feats} features vs {n_rows} rows)"
    print(f"\n[CHECK] {n_feats} features vs {n_rows} CSV rows -> {match}")

    # --- FAISS self-search ---
    print(f"\n[FAISS TEST] Tu search frame 0 trong chinh no...")
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    D, I = index.search(features[0:1], k=5)
    print(f"  Top-5 frame indices: {I[0]}")
    print(f"  Top-5 scores       : {D[0].round(4)}")
    print(f"  Ket qua tot nhat la chinh no (index=0): {'OK' if I[0][0] == 0 else 'LOI'}")

    # --- Map mau ---
    print(f"\n[MAP MAU] 3 keyframe dau:")
    for row in rows[:3]:
        print(f"  n={row['n']:>4} | pts_time={float(row['pts_time']):>10.3f}s | frame_idx={row['frame_idx']}")

    return features, rows, index


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load & Build BTC AIC 2026 FAISS Index")
    parser.add_argument("--mode",
                        choices=["inspect", "build", "build-self", "test-single"],
                        default="test-single",
                        help=(
                            "inspect    : kiem tra tat ca video trong BTC dir\n"
                            "build      : xay FAISS tu data BTC (data/btc/)\n"
                            "build-self : xay FAISS tu data TU TAO (data/processed/) [dung sau khi gop nhieu may]\n"
                            "test-single: test nhanh 1 cap npy+csv"
                        ))
    parser.add_argument("--npy",          type=str, default=None)
    parser.add_argument("--csv",          type=str, default=None)
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR))
    parser.add_argument("--map-dir",      type=str, default=str(MAP_DIR))
    args = parser.parse_args()

    if args.mode == "test-single":
        npy = args.npy or str(PROJECT_ROOT / "L21_V002.npy")
        csv_ = args.csv or str(PROJECT_ROOT / "L21_V001.csv")
        print(f"[INFO] Dung files mac dinh: {npy} + {csv_}")
        quick_test_single(npy, csv_)

    elif args.mode == "inspect":
        inspect_all_videos(
            features_dir=Path(args.features_dir),
            map_dir=Path(args.map_dir),
        )

    elif args.mode == "build":
        build_btc_index(
            features_dir=Path(args.features_dir),
            map_dir=Path(args.map_dir),
        )

    elif args.mode == "build-self":
        # Build FAISS tu data TU TAO cua pipeline (khong phai data BTC)
        # Dung sau khi nhieu may da encode CLIP per-video va gop .npy ve 1 may
        PROCESSED = OUTPUT_DIR
        SELF_FEAT_DIR = PROCESSED / "clip-features"     # data/processed/clip-features/
        SELF_MAP_DIR  = PROCESSED / "map-keyframes"     # data/processed/map-keyframes/
        SELF_OUT_FAISS   = PROCESSED / "index_self.faiss"
        SELF_OUT_MAP     = PROCESSED / "frame_index_self.json"
        SELF_OUT_FEATURES = PROCESSED / "clip_features_self.npy"

        # Tự động tách unknown.npy theo các file CSV nếu có
        unknown_file = SELF_FEAT_DIR / "unknown.npy"
        if unknown_file.exists():
            print("[INFO] Tự động tách unknown.npy theo danh sách video CSV...")
            u_data = np.load(unknown_file)
            csv_files = sorted(SELF_MAP_DIR.glob("*.csv"))
            offset = 0
            for csv_file in csv_files:
                vid = csv_file.stem
                df_count = sum(1 for _ in open(csv_file, encoding="utf-8")) - 1
                if df_count > 0 and offset + df_count <= len(u_data):
                    sub_feats = u_data[offset : offset + df_count]
                    np.save(str(SELF_FEAT_DIR / f"{vid}.npy"), sub_feats)
                    print(f"  -> Tách {vid}.npy: shape {sub_feats.shape}")
                    offset += df_count
            unknown_file.unlink()

        if not SELF_FEAT_DIR.exists() or not any(SELF_FEAT_DIR.glob("*.npy")):
            print(f"[LOI] Khong tim thay .npy trong {SELF_FEAT_DIR}")
            print("  Hay chay: python -m src.indexing.indexer --source self --per-video truoc")
        else:
            build_btc_index(
                features_dir=SELF_FEAT_DIR,
                map_dir=SELF_MAP_DIR,
                output_faiss=SELF_OUT_FAISS,
                output_map=SELF_OUT_MAP,
                output_features=SELF_OUT_FEATURES,
                load_objects=True,
                load_media=True,
            )
            print()
            print("Da xong! San sang retrieval:")
            print(f"  FAISS : {SELF_OUT_FAISS}")
            print(f"  Map   : {SELF_OUT_MAP}")
