"""
ingest.py — Task A2: Trich xuat Keyframe + Metadata
=====================================================
Output tuong thich voi format BTC:
  data/processed/keyframes/<video_id>/0000.jpg, 0001.jpg, ...
  data/processed/map-keyframes/<video_id>.csv   (cot: n, pts_time, fps, frame_idx)

Chay:
  # Voi data that (thi chinh thuc - BTC chi gui video):
  python -m src.indexing.ingest --source raw

  # Voi data mau thu:
  python -m src.indexing.ingest --source sample

  # Tuy chinh thu muc:
  python -m src.indexing.ingest --video-dir /duong/dan/thu/muc/video
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import csv
import os
import json
import argparse
from pathlib import Path


# ============================================================
# DUONG DAN GOC
# ============================================================
BASE_DIR     = Path(__file__).resolve().parent.parent.parent  # goc du an
DATA_DIR     = BASE_DIR / "data"
PROCESSED    = DATA_DIR / "processed"

# Thu muc dau vao (co the override bang --video-dir)
SAMPLE_DIR   = DATA_DIR / "sample_videos"
RAW_DIR      = DATA_DIR / "raw_videos"

# Thu muc dau ra (tuong thich format BTC)
KF_OUTPUT    = PROCESSED / "keyframes"        # data/processed/keyframes/<video_id>/xxxx.jpg
MAP_OUTPUT   = PROCESSED / "map-keyframes"    # data/processed/map-keyframes/<video_id>.csv
MEDIA_OUTPUT = PROCESSED / "media-info"       # data/processed/media-info/<video_id>.json  ← TU TAO
META_OUTPUT  = PROCESSED / "metadata.json"    # tuong thich nguoc voi code cu


# ============================================================
# TRICH KEYFRAME TU 1 VIDEO
# ============================================================

def extract_keyframes(video_path: Path, video_id: str,
                      fps_step: int = 1) -> list:
    """
    Trich keyframe tu video, luu theo format BTC:
      output/<video_id>/0000.jpg, 0001.jpg, ...

    Args:
        video_path : duong dan file video
        video_id   : ten video (khong co .mp4), vi du "L21_V001"
        fps_step   : lay 1 frame moi N giay (default 1)

    Returns:
        list of dict: moi phan tu la 1 keyframe voi cac truong:
          n, pts_time, fps, frame_idx  (khop chinh xac format CSV cua BTC)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [LOI] Khong mo duoc: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if fps == 0:
        print(f"  [LOI] FPS=0, bo qua: {video_path}")
        cap.release()
        return []

    print(f"  Video : {video_path.name}")
    print(f"  FPS   : {fps:.2f} | Tong frames: {total_frames} | Duration: {duration:.1f}s")

    # Tao thu muc output theo tung video (format BTC)
    out_dir = KF_OUTPUT / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, int(fps * fps_step))   # khoang cach frame giua moi lan lay

    # Danh sach cac frame_idx can lay (seek truc tiep, nhanh hon ~10x so voi doc tuan tu)
    target_frames = list(range(0, total_frames, step))

    keyframe_rows = []
    for n_idx, frame_idx in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        n = n_idx + 1
        pts_time  = frame_idx / fps
        frame_name = f"{n - 1:04d}.jpg"
        cv2.imwrite(str(out_dir / frame_name), frame)

        keyframe_rows.append({
            "n":         n,
            "pts_time":  round(pts_time, 4),
            "fps":       round(fps, 2),
            "frame_idx": frame_idx,
        })

    cap.release()
    print(f"  Da trich: {len(keyframe_rows)} keyframes -> {out_dir}")
    return keyframe_rows


# ============================================================
# LUU CSV (format BTC) + METADATA JSON (tuong thich code cu)
# ============================================================

def save_csv(video_id: str, keyframe_rows: list):
    """Luu file CSV theo format BTC: n, pts_time, fps, frame_idx."""
    MAP_OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = MAP_OUTPUT / f"{video_id}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(keyframe_rows)
    print(f"  CSV      : {csv_path}")
    return csv_path


def save_media_info(video_path: Path, video_id: str,
                    keyframe_rows: list) -> dict:
    """
    Tu trich xuat media info tu video bang cv2 (khong can BTC cung cap).
    Thay the cho file media-info/<video_id>.json cua BTC.
    Luu: data/processed/media-info/<video_id>.json

    Cac truong:
      fps, total_frames, duration (giay), width, height, n_keyframes
    """
    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = round(total_frames / fps, 4) if fps > 0 else 0.0

    info = {
        "video_id":     video_id,
        "fps":          round(fps, 4),
        "total_frames": total_frames,
        "duration":     duration,           # giay
        "width":        width,
        "height":       height,
        "n_keyframes":  len(keyframe_rows),
        "title":        "",                 # khong co YouTube title khi tu lam
        "description":  "",
    }

    MEDIA_OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = MEDIA_OUTPUT / f"{video_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Media    : {out_path}")
    return info


def update_metadata_json(video_id: str, keyframe_rows: list, metadata_dict: dict):
    """
    Cap nhat metadata.json tuong thich nguoc voi code cu (preprocess.py, indexer.py).
    Key = "<video_id>/<n-1:04d>.jpg"
    """
    for row in keyframe_rows:
        frame_name = f"{video_id}/{row['n'] - 1:04d}.jpg"
        metadata_dict[frame_name] = {
            "video_id":  video_id,
            "timestamp": row["pts_time"],
            "fps":       row["fps"],
            "frame_idx": row["frame_idx"],
        }


# ============================================================
# PIPELINE CHINH
# ============================================================

def run_ingest(video_dir: Path, fps_step: int = 1):
    """Chay pipeline trich keyframe toan bo thu muc video."""
    video_files = sorted(video_dir.glob("*.mp4")) + \
                  sorted(video_dir.glob("*.avi")) + \
                  sorted(video_dir.glob("*.mov")) + \
                  sorted(video_dir.glob("*.mkv"))

    if not video_files:
        print(f"Khong tim thay video trong: {video_dir}")
        return

    print(f"\n{'='*55}")
    print(f"  INGEST: {len(video_files)} videos tu {video_dir.name}/")
    print(f"  Lay 1 frame moi {fps_step}s")
    print(f"  Output: {KF_OUTPUT}")
    print(f"{'='*55}\n")

    # Doc metadata cu (neu co)
    metadata_dict = {}
    if META_OUTPUT.exists():
        with open(META_OUTPUT, encoding="utf-8") as f:
            metadata_dict = json.load(f)

    total_frames = 0
    for i, vpath in enumerate(video_files):
        video_id = vpath.stem  # "L21_V001", "video7014",...
        print(f"[{i+1}/{len(video_files)}] {vpath.name}")

        rows = extract_keyframes(vpath, video_id, fps_step=fps_step)
        if not rows:
            continue

        save_csv(video_id, rows)
        update_metadata_json(video_id, rows, metadata_dict)
        save_media_info(vpath, video_id, rows)   # tu tao media info, khong can BTC
        total_frames += len(rows)
        print()

    # Luu metadata.json tong hop
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

    print(f"{'='*55}")
    print(f"INGEST DONE!")
    print(f"  Videos xu ly : {len(video_files)}")
    print(f"  Tong keyframe: {total_frames}")
    print(f"  Keyframes    : {KF_OUTPUT}")
    print(f"  CSV maps     : {MAP_OUTPUT}")
    print(f"  Media info   : {MEDIA_OUTPUT}  <- tu tao, thay the BTC media-info")
    print(f"  Metadata JSON: {META_OUTPUT}")
    print(f"{'='*55}")
    print()
    print("BUOC TIEP THEO:")
    print("  [Local - CPU] python -m src.indexing.preprocess")
    print("  [Colab - GPU] Upload keyframes/ len Drive -> chay indexer.py")
    print()
    print("OBJECTS (Faster R-CNN) -> Viec cua nguoi D:")
    print("  Chay object detection tren keyframes da trich -> luu vao data/processed/objects/")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2: Trich Keyframe theo format BTC")
    parser.add_argument("--source", choices=["raw", "sample"], default="raw",
                        help="raw: data that (raw_videos/) | sample: data thu (sample_videos/)")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Tuy chinh thu muc chua video (ghi de --source)")
    parser.add_argument("--fps-step", type=float, default=0.5,
                        help=(
                            "Lay 1 frame moi N giay (default: 0.5).\n"
                            "  0.5s -> khoang cach 15 frames (30fps) -> an toan cho TRAKE (window <10 frame)\n"
                            "  1.0s -> khoang cach 30 frames -> co the truot window ngan cua TRAKE\n"
                            "  2.0s -> nhanh hon nhung rui ro cao voi TRAKE"
                        ))
    args = parser.parse_args()

    if args.video_dir:
        video_dir = Path(args.video_dir)
    elif args.source == "sample":
        video_dir = SAMPLE_DIR
    else:
        video_dir = RAW_DIR

    run_ingest(video_dir, fps_step=args.fps_step)