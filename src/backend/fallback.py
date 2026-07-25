"""
Logic fallback cho lớp tích hợp máy chủ.

Khi tác nhân truy xuất (retrieval agent) gặp sự cố hoặc các mô-đun truy xuất không khả dụng, mô-đun này cung cấp một phương án dự phòng an toàn, chi phí thấp bằng cách sử dụng triển khai trình truy xuất đơn giản từ lớp truy xuất.
"""

from typing import Any, Dict, List


def run_fallback_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Return a basic fallback payload that still satisfies the API contract."""
    try:
        from src.retrieval.retriever import Retriever

        retriever = Retriever(use_dummy=True)
        raw_results = retriever.search_keyframes(query_text=query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - defensive path
        return {"segments": [], "error": str(exc)}

    segments: List[Dict[str, Any]] = []
    
    # Kiểm tra an toàn phòng trường hợp raw_results không phải là iterable (list)
    if not isinstance(raw_results, list):
        return {"segments": [], "error": "Invalid results format from retriever"}

    for idx, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue

        score = float(item.get("score") or 0.0)

        # Xử lý an toàn cho frame_idx phòng trường hợp None hoặc sai kiểu
        raw_frame_idx = item.get("frame_idx")
        try:
            frame_idx = float(raw_frame_idx) if raw_frame_idx is not None else float(idx)
        except (ValueError, TypeError):
            frame_idx = float(idx)

        segments.append(
            {
                "video_id": str(item.get("video_id") or "unknown"),
                "start": frame_idx,
                "end": frame_idx + 1.0,
                "avg_similarity": round(score, 4),
                "peak_similarity": round(score, 4),
                "n_frames": 1,
            }
        )

    return {"segments": segments, "error": None}