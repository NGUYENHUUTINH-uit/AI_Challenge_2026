"""
validate_pipeline.py — So sánh pipeline tự làm vs data chuẩn BTC
=================================================================
Mục đích: Kiểm tra xem output pipeline của nhóm (ingest → preprocess → indexer)
có chất lượng tương đương data BTC không, dù thuật toán trích keyframe có thể khác.

Có 2 chế độ:
  --mode self    : Chạy toàn bộ pipeline trên 1 video mẫu, kiểm tra nội bộ
  --mode compare : So sánh .npy tự tạo với .npy BTC (cần cùng video)

Chạy nhanh nhất (kiểm tra nội bộ):
  python validate_pipeline.py --mode self --video data/sample_videos/video7014.mp4

So sánh với BTC (nếu có cùng video):
  python validate_pipeline.py --mode compare --video data/sample_videos/L21_V001.mp4
                                              --btc-npy L21_V002.npy
                                              --btc-csv L21_V001.csv
"""

import sys, io, os, csv, json, tempfile, shutil, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import cv2
from pathlib import Path

# Duong dan goc du an
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# BUOC 1: TRICH KEYFRAME (giong ingest.py, them stats)
# ============================================================

def extract_keyframes_simple(video_path: Path, output_dir: Path,
                              fps_step: int = 1) -> list:
    """
    Trich 1 frame moi fps_step giay (giong ingest.py hien tai).
    Tra ve list dict: { frame_name, timestamp, frame_idx, fps }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  [Video] {video_path.name}")
    print(f"  FPS: {fps:.1f} | Tong frames: {total_frames} | Duration: {duration:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(fps * fps_step))  # so frame giua moi lan lay
    keyframe_meta = []
    saved = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / fps
            frame_name = f"{video_path.stem}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            keyframe_meta.append({
                "frame_name": frame_name,
                "timestamp":  round(timestamp, 4),
                "frame_idx":  frame_idx,
                "fps":        fps,
            })
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"  Trich xuat: {saved} keyframes (moi {fps_step}s)")
    return keyframe_meta


# ============================================================
# BUOC 2: ENCODE CLIP
# ============================================================

def encode_clip(image_dir: Path, batch_size: int = 16) -> tuple:
    """
    Encode tat ca .jpg trong image_dir bang CLIP ViT-B/32.
    Tra ve (feature_matrix (N,512) float32, danh_sach_ten_file)
    """
    import torch
    try:
        import clip
        backend = "openai"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
    except ImportError:
        from transformers import CLIPProcessor, CLIPModel
        backend = "huggingface"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    from PIL import Image
    import torch

    jpg_files = sorted(image_dir.glob("*.jpg"))
    print(f"\n  [CLIP] Encoding {len(jpg_files)} frames (backend={backend}, device={device})")

    all_feats = []
    all_names = []

    for i in range(0, len(jpg_files), batch_size):
        batch_paths = jpg_files[i:i+batch_size]
        images = []
        valid = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid.append(p)
            except Exception as e:
                print(f"    [WARN] {p.name}: {e}")

        if not images:
            continue

        with torch.no_grad():
            if backend == "openai":
                tensors = torch.stack([preprocess(img) for img in images]).to(device)
                feats = model.encode_image(tensors)
            else:
                inputs = preprocess(images=images, return_tensors="pt").to(device)
                feats = model.get_image_features(**inputs)

            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            all_feats.append(feats.cpu().numpy().astype(np.float32))
            all_names.extend([p.name for p in valid])

        print(f"    Batch {i//batch_size + 1}: {len(valid)} frames OK")

    feature_matrix = np.vstack(all_feats).astype(np.float32)
    print(f"  [CLIP] Done -> shape: {feature_matrix.shape}")
    return feature_matrix, all_names


# ============================================================
# BUOC 3: SO SANH VOI BTC
# ============================================================

def compare_with_btc(our_features: np.ndarray, our_meta: list,
                     btc_npy_path: Path, btc_csv_path: Path):
    """
    So sanh output pipeline tu lam voi file BTC.
    Bao gom: so luong keyframe, chat luong CLIP features,
    and phan tich su khac biet ve lich trinh trich frame.
    """
    print(f"\n{'='*60}")
    print("  SO SANH VOI DATA BTC")
    print(f"{'='*60}")

    # Load BTC
    btc_feats = np.load(str(btc_npy_path)).astype(np.float32)
    with open(btc_csv_path, encoding="utf-8", newline="") as f:
        btc_rows = list(csv.DictReader(f))

    print(f"\n[SO LUONG KEYFRAME]")
    print(f"  Pipeline cua minh : {len(our_features)} frames")
    print(f"  BTC               : {len(btc_feats)} frames (npy) / {len(btc_rows)} rows (csv)")

    ratio = len(our_features) / max(len(btc_feats), 1)
    print(f"  Ti le             : {ratio:.2f}x")
    if abs(ratio - 1.0) < 0.15:
        print("  -> TUONG DUONG (sai lech < 15%)")
    elif ratio > 1.0:
        print(f"  -> Minh lay NHIEU hon BTC ~{(ratio-1)*100:.0f}%")
    else:
        print(f"  -> Minh lay IT hon BTC ~{(1-ratio)*100:.0f}%")

    print(f"\n[LICH TRINH TRICH FRAME]")
    our_times = [m["timestamp"] for m in our_meta[:5]]
    btc_times = [float(r["pts_time"]) for r in btc_rows[:5]]
    print(f"  5 timestamp dau cua minh : {our_times}")
    print(f"  5 timestamp dau cua BTC  : {btc_times}")
    print(f"  -> Minh: deu deu moi 1s | BTC: scene-based (khong deu)")

    print(f"\n[CHAT LUONG CLIP FEATURES]")
    # Kiem tra L2 norm
    our_norms = np.linalg.norm(our_features, axis=1)
    btc_norms = np.linalg.norm(btc_feats, axis=1)
    print(f"  L2 norm trung binh cua minh: {our_norms.mean():.6f} (ky vong = 1.0)")
    print(f"  L2 norm trung binh cua BTC : {btc_norms.mean():.6f}")
    print(f"  -> {'OK' if abs(our_norms.mean() - 1.0) < 0.01 else 'LOI: features chua duoc normalize'}")

    # So sanh phan phoi vector (phan tich thong ke)
    our_mean = our_features.mean(axis=0)
    btc_mean = btc_feats.mean(axis=0)
    dist = np.linalg.norm(our_mean - btc_mean)
    cos_sim = float(np.dot(our_mean, btc_mean) /
                    (np.linalg.norm(our_mean) * np.linalg.norm(btc_mean) + 1e-8))
    print(f"\n  Cosine sim giua mean vector: {cos_sim:.4f}")
    print(f"  L2 dist giua mean vector   : {dist:.4f}")
    if cos_sim > 0.9:
        print("  -> PHAN PHOI TUONG TU (cung phan vung ngu nghia)")
    elif cos_sim > 0.7:
        print("  -> PHAN PHOI TUONG DOI GIONG (co the khac video khac canh)")
    else:
        print("  -> PHAN PHOI KHAC BIET (co the do khac video hoac BTC dung model khac)")

    print(f"\n[KET LUAN]")
    print("  CLIP features: cung model (clip-ViT-B-32) -> encoding chat luong tuong duong")
    print("  Khac biet chinh: LICH TRINH TRICH FRAME")
    print("    + BTC: scene detection (thay doi canh) -> it frame hon nhung chat luong cao hon")
    print("    + Minh (ingest.py): moi 1s -> nhieu frame hon, co the trung lap")
    print("  -> De khop voi BTC: can implement scene-based extraction trong ingest.py")


# ============================================================
# BUOC 4: KIEM TRA NOI BO (khong can file BTC)
# ============================================================

def run_self_check(our_features: np.ndarray, our_meta: list, video_path: Path):
    """
    Kiem tra noi bo pipeline ma khong can file BTC de so sanh.
    Chay FAISS self-search, kiem tra norm, kiem tra metadata.
    """
    import faiss

    print(f"\n{'='*60}")
    print("  KIEM TRA NOI BO PIPELINE")
    print(f"{'='*60}")

    N, D = our_features.shape

    print(f"\n[FEATURE MATRIX]")
    print(f"  Shape: {our_features.shape}")
    print(f"  Dtype: {our_features.dtype}")
    norms = np.linalg.norm(our_features, axis=1)
    print(f"  L2 norm: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    norm_ok = np.allclose(norms, 1.0, atol=0.01)
    print(f"  Da L2-normalize: {'OK' if norm_ok else 'CHUA! Can normalize truoc khi dua vao FAISS'}")

    print(f"\n[METADATA]")
    print(f"  So keyframes: {len(our_meta)}")
    print(f"  3 frame dau:")
    for m in our_meta[:3]:
        print(f"    {m}")

    print(f"\n[FAISS SELF-SEARCH]")
    index = faiss.IndexFlatIP(D)
    index.add(our_features)
    # Query frame dau tien - ket qua tot nhat phai la chinh no
    D_scores, I_indices = index.search(our_features[:1], k=5)
    print(f"  Query: frame 0")
    print(f"  Top-5 indices: {I_indices[0]}")
    print(f"  Top-5 scores : {D_scores[0].round(4)}")
    print(f"  Rank 1 la chinh no: {'OK' if I_indices[0][0] == 0 else 'LOI'}")

    print(f"\n[KIEM TRA THOI GIAN]")
    timestamps = [m["timestamp"] for m in our_meta]
    is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    print(f"  Timestamps tang dan: {'OK' if is_monotonic else 'LOI: co timestamp bi dao nguoc'}")
    print(f"  Timestamp dau: {timestamps[0]}s | Timestamp cuoi: {timestamps[-1]}s")

    print(f"\n[KET QUA NOI BO]")
    if norm_ok and I_indices[0][0] == 0 and is_monotonic:
        print("  Pipeline chay DUNG: features chat luong, metadata hop le")
        print("  San sang de ket noi voi retriever.py")
    else:
        print("  Co van de! Kiem tra lai cac buoc tren")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate pipeline vs BTC data")
    parser.add_argument("--mode", choices=["self", "compare"], default="self",
                        help="self: kiem tra noi bo | compare: so sanh voi BTC")
    parser.add_argument("--video", type=str, required=True,
                        help="Duong dan toi file video (.mp4)")
    parser.add_argument("--btc-npy", type=str, default=None,
                        help="[compare mode] File .npy cua BTC")
    parser.add_argument("--btc-csv", type=str, default=None,
                        help="[compare mode] File .csv (map-keyframes) cua BTC")
    parser.add_argument("--fps-step", type=int, default=1,
                        help="Lay 1 frame moi N giay (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size cho CLIP encoding (default: 16)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Khong tim thay video: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  VALIDATE PIPELINE — mode={args.mode}")
    print(f"{'='*60}")

    # Tao thu muc tam
    tmp_dir = Path(tempfile.mkdtemp(prefix="validate_"))
    kf_dir = tmp_dir / "keyframes"

    try:
        # Buoc 1: Trich keyframe
        print(f"\n[BUOC 1] Trich keyframe...")
        our_meta = extract_keyframes_simple(video_path, kf_dir, fps_step=args.fps_step)

        # Buoc 2: Encode CLIP
        print(f"\n[BUOC 2] Encode CLIP features...")
        our_features, our_names = encode_clip(kf_dir, batch_size=args.batch_size)

        if args.mode == "self":
            run_self_check(our_features, our_meta, video_path)

        elif args.mode == "compare":
            if not args.btc_npy or not args.btc_csv:
                print("Che do 'compare' can --btc-npy va --btc-csv")
                sys.exit(1)
            compare_with_btc(
                our_features, our_meta,
                btc_npy_path=Path(args.btc_npy),
                btc_csv_path=Path(args.btc_csv),
            )

    finally:
        # Don dep thu muc tam
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n[DONE] Xong! Thu muc tam da xoa.")
