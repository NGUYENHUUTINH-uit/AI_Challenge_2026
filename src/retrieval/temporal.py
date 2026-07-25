"""
temporal.py — Task B4: Temporal Localization
Định vị các đoạn thời gian liên quan trong video dựa trên similarity score
của keyframes, KHÔNG dùng cửa sổ thời gian cố định (theo đúng yêu cầu đề bài).

Thuật toán: Similarity Thresholding (ngưỡng động theo phân phối điểm) +
Temporal Grouping (gom cụm các frame liên tiếp trong cùng video).
"""
from dataclasses import dataclass, field
from itertools import groupby
from typing import Callable, Optional
import statistics


@dataclass
class TemporalConfig:
    """Toàn bộ tham số có thể tinh chỉnh .
    Khi hiệu chỉnh trên data thật (giai đoạn 29/7-2/8), chỉ cần đổi giá trị ở đây,
    không phải sửa thuật toán.
    """
    base_threshold: float = 0.55          # sàn an toàn, không bao giờ thấp hơn mức này
    gap_tolerance: float = 2.0            # giây — khoảng cách tối đa để coi là cùng 1 đoạn
    threshold_method: str = "percentile"  # "percentile" | "mean" | "gap_based"
    percentile: float = 80.0              # dùng khi threshold_method = "percentile"
    mean_ratio: float = 0.9               # dùng khi threshold_method = "mean"
    min_segment_frames: int = 1           # lọc bỏ đoạn quá ngắn (nhiễu 1 frame lẻ)


def _percentile(data: list[float], pct: float) -> float:
    """Percentile thủ công, không phụ thuộc numpy (giảm dependency)."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def _compute_dynamic_threshold(scores: list[float], cfg: TemporalConfig) -> float:
    """
    3 chiến lược tính ngưỡng động — lý do cần nhiều lựa chọn:

    - "mean": đơn giản, nhanh, nhưng có thể rỗng nếu query khó (mean thấp
      + base_threshold ép sàn cao hơn mean).
    - "percentile": LUÔN giữ lại top X% điểm cao nhất trong chính tập kết quả
      này — không bao giờ rỗng, tự thích nghi độ khó của từng query. Khuyến
      nghị dùng mặc định.
    - "gap_based": tìm khoảng cách (gap) lớn nhất giữa 2 điểm liền kề khi sort
      giảm dần — ranh giới tự nhiên giữa "liên quan" và "nhiễu nền", không
      cần đoán trước tỉ lệ %. Tốt nhất khi phân phối điểm có 2 cụm rõ rệt
      (relevant cluster vs background), nhưng có thể kém ổn định khi dữ liệu
      đồng đều.
    """
    if not scores:
        return cfg.base_threshold

    if cfg.threshold_method == "mean":
        return max(cfg.base_threshold, statistics.mean(scores) * cfg.mean_ratio)

    if cfg.threshold_method == "percentile":
        return max(cfg.base_threshold, _percentile(scores, cfg.percentile))

    if cfg.threshold_method == "gap_based":
        s = sorted(scores, reverse=True)
        if len(s) < 2:
            return cfg.base_threshold
        gaps = [(s[i] - s[i + 1], i) for i in range(len(s) - 1)]
        biggest_gap_idx = max(gaps, key=lambda x: x[0])[1]
        candidate = s[biggest_gap_idx]
        return max(cfg.base_threshold, candidate)

    raise ValueError(f"threshold_method không hợp lệ: {cfg.threshold_method}")


def _segment_single_video(frames: list[dict], cfg: TemporalConfig) -> list[dict]:
    """Gom cụm cho MỘT video (frames đã sort theo timestamp, cùng video_id)."""
    scores = [f["similarity"] for f in frames]
    threshold = _compute_dynamic_threshold(scores, cfg)
    relevant = [f for f in frames if f["similarity"] >= threshold]
    if not relevant:
        return []

    segments: list[list[dict]] = []
    current = [relevant[0]]
    for f in relevant[1:]:
        if f["timestamp"] - current[-1]["timestamp"] <= cfg.gap_tolerance:
            current.append(f)
        else:
            segments.append(current)
            current = [f]
    segments.append(current)

    results = []
    for seg in segments:
        if len(seg) < cfg.min_segment_frames:
            continue
        results.append({
            "start": seg[0]["timestamp"],
            "end": seg[-1]["timestamp"],
            "avg_similarity": round(sum(x["similarity"] for x in seg) / len(seg), 4),
            "peak_similarity": round(max(x["similarity"] for x in seg), 4),
            "n_frames": len(seg),
        })
    return results


def find_time_range(
    keyframes: list[dict],
    config: Optional[TemporalConfig] = None,
) -> list[dict]:
    """
    Entry point chính của B4.

    Input:
        keyframes: list[{"video_id": str, "timestamp": float, "similarity": float}]
                   (kết quả thô trả về từ retriever.search_keyframes())
        config: TemporalConfig — nếu None, dùng mặc định.

    Output:
        list[{"video_id", "start", "end", "avg_similarity", "peak_similarity", "n_frames"}]
        Đã SẮP XẾP giảm dần theo peak_similarity — đoạn liên quan nhất lên đầu.

    Quan trọng: TÁCH RIÊNG theo video_id trước khi gom cụm — bug phổ biến nếu
    bỏ qua bước này là gom nhầm frame của 2 video khác nhau thành 1 đoạn.
    """
    cfg = config or TemporalConfig()
    if not keyframes:
        return []

    sorted_frames = sorted(keyframes, key=lambda x: (x["video_id"], x["timestamp"]))

    all_segments = []
    for video_id, group in groupby(sorted_frames, key=lambda x: x["video_id"]):
        segs = _segment_single_video(list(group), cfg)
        for s in segs:
            s["video_id"] = video_id
        all_segments.extend(segs)

    return sorted(all_segments, key=lambda x: -x["peak_similarity"])


def evaluate_iou(predicted: dict, ground_truth: tuple[float, float]) -> float:
    """
    Đo Intersection-over-Union giữa 1 đoạn dự đoán và đáp án đúng.
    Dùng cho giai đoạn hiệu chỉnh threshold trên data thật (29/7-2/8).
    """
    gt_start, gt_end = ground_truth
    inter = max(0.0, min(predicted["end"], gt_end) - max(predicted["start"], gt_start))
    union = (predicted["end"] - predicted["start"]) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0