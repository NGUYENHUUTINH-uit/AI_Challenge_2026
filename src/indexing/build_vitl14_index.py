"""
build_vitl14_index.py — Build CLIP ViT-L/14 FAISS Index cho AIC 2026
=====================================================================
Chạy trên Colab (GPU T4/A100) theo pattern tương tự detect_objects.py:

  1. Clone repo từ GitHub (có GITHUB_TOKEN)
  2. Mount Google Drive (data keyframes đã có sẵn từ bước detect objects)
  3. Chạy script này: python src/indexing/build_vitl14_index.py
  4. File index_btc_vitl14.faiss tự động lưu về Drive

Kết quả:
  /content/drive/MyDrive/AIC2026/index_btc_vitl14.faiss  (~540 MB)
  /content/drive/MyDrive/AIC2026/frame_index_btc.json    (frame map, giữ nguyên)

Sau khi download về local:
  data/processed/index_btc_vitl14.faiss  ← retriever tự dùng (ưu tiên cao nhất)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Build CLIP ViT-L/14 FAISS index")
    p.add_argument("--keyframes-dir", default="data/btc/Keyframes",
                   help="Thư mục gốc chứa keyframes (default: data/btc/Keyframes)")
    p.add_argument("--frame-map", default="data/processed/frame_index_btc.json",
                   help="Frame map JSON từ bước build-btc")
    p.add_argument("--output", default="data/processed/index_btc_vitl14.faiss",
                   help="Đường dẫn file output FAISS index")
    p.add_argument("--drive-output", default=None,
                   help="Đồng thời lưu vào Google Drive (vd: /content/drive/MyDrive/AIC2026/)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size encoding ảnh (256 cho T4, 512 cho A100)")
    p.add_argument("--model", default="ViT-L/14",
                   choices=["ViT-L/14", "ViT-B/32", "ViT-B/16", "ViT-L/14@336px"],
                   help="CLIP model variant")
    p.add_argument("--resume", action="store_true",
                   help="Tiếp tục từ file partial nếu có (tránh encode lại từ đầu)")
    return p.parse_args()


def build_index(args):
    import torch
    import faiss
    import clip
    from PIL import Image

    # === Setup ===
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Build CLIP {args.model} FAISS Index — AIC 2026")
    print("=" * 60)
    print(f"Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  WARNING: Chạy trên CPU sẽ RẤT CHẬM (~24h cho 177K frames)")
        print("    Khuyến nghị: dùng Colab với GPU T4/A100")

    # === Load model ===
    print(f"\n📥 Loading CLIP {args.model}...")
    model, preprocess = clip.load(args.model, device=DEVICE)
    model.eval()
    DIM = model.visual.output_dim
    print(f"✅ Model loaded! Output dim: {DIM}")

    # === Load frame map ===
    frame_map_path = Path(args.frame_map)
    if not frame_map_path.exists():
        print(f"❌ Không tìm thấy frame map: {frame_map_path}")
        print("   Hãy chạy: python run.py --stage build-btc trước")
        sys.exit(1)

    print(f"\n📥 Loading frame map: {frame_map_path}")
    with open(frame_map_path, "r", encoding="utf-8") as f:
        frame_map = json.load(f)

    sorted_entries = sorted(frame_map.items(), key=lambda x: int(x[0]))
    total_frames = len(sorted_entries)
    print(f"✅ {total_frames} frames")

    # === Resume: load partial embeddings nếu có ===
    partial_path = Path(args.output + ".partial.npy")
    resume_from = 0
    all_embeddings = np.zeros((total_frames, DIM), dtype=np.float32)

    if args.resume and partial_path.exists():
        saved = np.load(partial_path)
        resume_from = saved.shape[0]
        all_embeddings[:resume_from] = saved
        print(f"♻️  Resume từ frame {resume_from}/{total_frames}")

    # === Encode ===
    kf_dir = Path(args.keyframes_dir)
    errors = 0
    start_time = time.time()

    print(f"\n🚀 Encoding frames {resume_from} → {total_frames} (batch={args.batch_size})...")

    for batch_start in tqdm(range(resume_from, total_frames, args.batch_size), desc="Encoding"):
        batch_entries = sorted_entries[batch_start:batch_start + args.batch_size]
        images = []

        for faiss_idx, info in batch_entries:
            video_id = info.get("video_id", "")
            filename = info.get("filename", "")
            n_val = info.get("n")

            if not filename and n_val is not None:
                filename = f"{int(n_val):03d}.jpg"

            img_path = kf_dir / video_id / Path(filename).name

            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
            except Exception:
                errors += 1
                img = preprocess(Image.new("RGB", (224, 224), (128, 128, 128)))
            images.append(img)

        if images:
            batch_tensor = torch.stack(images).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

            feats_np = feats.cpu().numpy().astype(np.float32)
            end = min(batch_start + len(images), total_frames)
            all_embeddings[batch_start:end] = feats_np

        # Lưu checkpoint mỗi 5000 frames (tránh mất dữ liệu nếu Colab crash)
        processed_so_far = batch_start + len(images)
        if processed_so_far % 5000 < args.batch_size:
            np.save(partial_path, all_embeddings[:processed_so_far])
            elapsed = time.time() - start_time
            eta = (total_frames - processed_so_far) / max(processed_so_far, 1) * elapsed
            print(f"\n💾 Checkpoint: {processed_so_far}/{total_frames} frames "
                  f"| elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    elapsed = time.time() - start_time
    print(f"\n✅ Encoding done! {elapsed/60:.1f}m | Errors (missing files): {errors}")

    # === Build FAISS index ===
    print("\n📦 Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(DIM)
    index.add(all_embeddings)
    print(f"✅ Index: {index.ntotal} vectors, dim={index.d}")

    # === Save ===
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n💾 Saved: {out_path} ({size_mb:.1f} MB)")

    # Xóa checkpoint partial
    if partial_path.exists():
        partial_path.unlink()

    # === Copy sang Drive nếu có ===
    if args.drive_output:
        import shutil
        drive_dir = Path(args.drive_output)
        drive_dir.mkdir(parents=True, exist_ok=True)

        dest_index = drive_dir / out_path.name
        shutil.copy(out_path, dest_index)
        print(f"☁️  Copied to Drive: {dest_index} ({size_mb:.1f} MB)")

        # Copy frame map đi kèm
        fm_dest = drive_dir / frame_map_path.name
        shutil.copy(frame_map_path, fm_dest)
        print(f"☁️  Copied frame map: {fm_dest}")

    # === Quick verify ===
    print("\n🧪 Quick verify — test search...")
    try:
        test_q = "cycling race cyclists colorful jerseys"
        tokens = clip.tokenize([test_q], truncate=True).to(DEVICE)
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores, indices = index.search(text_feat.cpu().numpy().astype(np.float32), 3)
        print(f"Query: '{test_q}'")
        for rank, (sc, idx) in enumerate(zip(scores[0], indices[0])):
            info = frame_map.get(str(idx), {})
            print(f"  #{rank+1}: {info.get('video_id')}/{info.get('filename')} score={sc:.4f}")
    except Exception as e:
        print(f"⚠️ Verify failed: {e}")

    print(f"\n✅ DONE! Khi tải file về local, đặt vào:")
    print(f"   data/processed/{out_path.name}")
    print("   Retriever sẽ tự dùng ViT-L/14 (ưu tiên cao hơn ViT-B/32)")


if __name__ == "__main__":
    args = parse_args()
    build_index(args)
