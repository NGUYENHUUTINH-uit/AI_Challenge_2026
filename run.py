"""
run.py — Điểm đầu vào đơn giản hóa cho Pipeline AIC 2026
=====================================================
Gói gọn tất cả các câu lệnh module để chạy dễ dàng hơn.

VÒNG SƠ TUYỂN (dùng data BTC gửi sẵn):
    python run.py --stage build-btc

VÒNG CHÍNH THỨC (BTC chỉ gửi video):
    python run.py --stage ingest
    python run.py --stage detect
    python run.py --stage encode
    python run.py --stage build-self

TEST NHANH TOÀN BỘ (trên máy local):
    python run.py --stage all --source sample
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list, description: str):
    """Chạy lệnh và in kết quả."""
    print(f"\n{'='*60}")
    print(f"  [{description}]")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[LỖI] '{description}' thất bại với mã lỗi {result.returncode}")
        sys.exit(result.returncode)


def stage_ingest(args):
    """Bước 1: Trích keyframe + CSV + media-info từ video."""
    cmd = [sys.executable, "-m", "src.indexing.ingest",
           "--source", args.source,
           "--fps-step", str(args.fps_step)]
    run_cmd(cmd, "TRÍCH KEYFRAME (ingest.py)")


def stage_detect(args):
    """Bước 2: Object detection trên keyframe."""
    cmd = [sys.executable, "-m", "src.indexing.detect_objects",
           "--model", args.obj_model,
           "--source", "self"]
    run_cmd(cmd, f"OBJECT DETECTION (model={args.obj_model})")


def stage_encode(args):
    """Bước 3: CLIP encoding (nên chạy trên Colab GPU)."""
    cmd = [sys.executable, "-m", "src.indexing.indexer",
           "--source", "self",
           "--per-video",
           "--batch-size", str(args.batch_size)]
    run_cmd(cmd, "CLIP ENCODING (indexer.py --per-video)")


def stage_filter(args):
    """Tiền xử lý: Lọc frame mờ/tối/sáng quá mức (preprocess.py)."""
    cmd = [sys.executable, "-m", "src.indexing.preprocess"]
    run_cmd(cmd, "LỌC FRAME NHIỄU (preprocess.py)")


def stage_build_self(args):
    """Bước 4: Build FAISS từ data tự tạo (sau khi gom tất cả máy)."""
    cmd = [sys.executable, "-m", "src.indexing.load_btc_data",
           "--mode", "build-self"]
    run_cmd(cmd, "BUILD FAISS TỪ DATA TỰ TẠO (build-self)")


def stage_build_btc(args):
    """Vòng sơ tuyển: Build FAISS từ data BTC cấp sẵn."""
    cmd = [sys.executable, "-m", "src.indexing.load_btc_data",
           "--mode", "build"]
    run_cmd(cmd, "BUILD FAISS TỪ DATA BTC (build)")


def main():
    parser = argparse.ArgumentParser(
        description="AIC 2026 Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["ingest", "filter", "preprocess", "group-scenes", "group_scenes", "detect", "encode", "build-self", "build-btc", "all"],
        help=(
            "Chọn giai đoạn cần chạy:\n"
            "  ingest       : Trích keyframe từ video (Local CPU)\n"
            "  filter       : Lọc frame mờ / quá tối / quá sáng (preprocess.py)\n"
            "  group-scenes : Tiền xử lý & gom cụm phân cảnh (preprocess.py)\n"
            "  detect       : Object detection trên keyframes (Local CPU)\n"
            "  encode       : CLIP encoding (khuyên dùng trên Colab GPU)\n"
            "  build-self   : Build FAISS từ data tự tạo (Local, sau Colab)\n"
            "  build-btc    : Build FAISS từ data BTC cấp sẵn (Vòng sơ tuyển)\n"
            "  all          : Chạy tất cả từ đầu đến cuối (Local test)"
        )
    )

    parser.add_argument(
        "--source",
        choices=["raw", "sample"],
        default="raw",
        help="raw: data/raw_videos/ | sample: data/sample_videos/"
    )

    parser.add_argument(
        "--fps-step", type=float, default=0.5,
        help="Lấy 1 frame mỗi N giây (default: 0.5s)"
    )

    parser.add_argument(
        "--obj-model",
        choices=["coco", "openimages", "grounding-dino"],
        default="coco",
        help="Model object detection (default: coco)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size cho CLIP encoding (default: 32, Colab GPU: 64)"
    )

    args = parser.parse_args()

    if args.stage == "ingest":
        stage_ingest(args)

    elif args.stage in ["filter", "preprocess", "group-scenes", "group_scenes"]:
        stage_filter(args)

    elif args.stage == "detect":
        stage_detect(args)

    elif args.stage == "encode":
        print("[INFO] Bước này nên chạy trên Colab GPU.")
        print("[INFO] Đang chạy trên máy hiện tại...")
        stage_encode(args)

    elif args.stage == "build-self":
        stage_build_self(args)

    elif args.stage == "build-btc":
        stage_build_btc(args)

    elif args.stage == "all":
        print("[INFO] Chạy toàn bộ pipeline trên Local (CPU)")
        stage_ingest(args)
        stage_detect(args)
        stage_encode(args)
        stage_build_self(args)

    print("\n" + "="*60)
    print(f"  DONE: stage={args.stage}")
    print("="*60)


if __name__ == "__main__":
    main()
