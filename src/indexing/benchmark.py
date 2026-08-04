"""
benchmark.py - Task A4: Đo đạc tốc độ Indexing
Đánh giá tốc độ trích xuất vector CLIP với các batch_size khác nhau.

Đặt file này cùng thư mục với indexer.py (src/indexing/).
"""

import csv
import time
from pathlib import Path
import sys

# Thêm thư mục gốc vào sys.path để import đúng
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.indexing.indexer import (
    load_clip_model,
    extract_batch_openai,
    extract_batch_huggingface,
    KEYFRAME_DIR,
    _sort_key,
)

# ============================================================
# CONFIG
# ============================================================
REPEAT_FACTOR = 7      # đủ lớn để đo ổn định
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128]   # thêm 64, 128 cho GPU test
N_TRIALS = 3            # số lần lặp mỗi batch_size, lấy min để loại nhiễu hệ thống


def _collect_benchmark_frames(keyframe_dir: Path) -> list:
    """Hỗ trợ cả flat và subfolder format."""
    sub = list(keyframe_dir.glob("*/*.jpg"))
    if sub:
        return sorted(sub, key=lambda p: (p.parent.name, p.name))
    return sorted(keyframe_dir.glob("*.jpg"))


def run_benchmark():
    print("=" * 50)
    print("  A4 - BENCHMARK INDEXING SPEED")
    print("=" * 50)

    # 1. Load model
    model, preprocessor, device, backend = load_clip_model()

    # 2. Lấy danh sách file ảnh (hỗ trợ flat + subfolder)
    base_paths = _collect_benchmark_frames(KEYFRAME_DIR)
    if not base_paths:
        print("No images found!")
        return

    # 3. Nhân bản DANH SÁCH đường dẫn (không tốn thêm disk) để có đủ khối lượng đo
    test_paths = base_paths * REPEAT_FACTOR
    total_frames = len(test_paths)
    print(f"\n[INFO] {len(base_paths)} ảnh gốc x {REPEAT_FACTOR} lần = {total_frames} lượt encode")
    print(f"[INFO] Backend: {backend} | Device: {device.upper()}\n")

    # 4. Warm-up: loại bỏ chi phí khởi động (torch kernel init, cache...)
    print("[INFO] Warming up...")
    warmup_batch = test_paths[:min(4, total_frames)]
    if backend == "openai":
        _ = extract_batch_openai(warmup_batch, model, preprocessor, device)
    else:
        _ = extract_batch_huggingface(warmup_batch, model, preprocessor, device)

    # 5. Benchmark từng batch_size
    print(f"{'Batch Size':<12} | {'Time (s)':<15} | {'Speed (FPS)':<15}")
    print("-" * 50)

    results = []
    for bs in BATCH_SIZES:
        durations = []
        for _trial in range(N_TRIALS):
            start_time = time.perf_counter()
            for i in range(0, total_frames, bs):
                batch = test_paths[i: i + bs]
                if backend == "openai":
                    _ = extract_batch_openai(batch, model, preprocessor, device)
                else:
                    _ = extract_batch_huggingface(batch, model, preprocessor, device)
            durations.append(time.perf_counter() - start_time)

        # min() thay vì trung bình: nhiễu hệ thống chỉ làm CHẬM thêm, không bao giờ
        # làm nhanh hơn baseline thật -> giá trị nhỏ nhất phản ánh tốc độ thật nhất
        duration = min(durations)
        fps = total_frames / duration
        results.append((bs, round(duration, 3), round(fps, 2)))
        print(f"{bs:<12} | {duration:<15.3f} | {fps:<15.2f}")

    print("-" * 50)

    # 6. Chọn batch_size tối ưu (FPS cao nhất)
    best = max(results, key=lambda r: r[2])
    print(f"\n[KẾT LUẬN] batch_size tối ưu: {best[0]} ({best[2]} FPS)")

    # Ước tính thời gian cho data thật
    for n_real in [10_000, 100_000, 500_000]:
        est_sec = n_real / best[2]
        print(f"  Ước tính encode {n_real:>7,} ảnh thật: ~{est_sec/60:.1f} phút")

    # 7. Lưu báo cáo CSV
    with open("benchmark_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "time_sec", "fps"])
        writer.writerows(results)
    print("\n[SAVE] benchmark_report.csv")
    print("Benchmark completed!")


if __name__ == "__main__":
    run_benchmark()