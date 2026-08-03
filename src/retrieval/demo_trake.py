"""
demo_trake.py — Test luồng TRAKE với mock decomposer + mock search_fn CÓ NGỮ
NGHĨA (similarity gắn đúng với timestamp mong đợi của từng sự kiện), để kiểm
chứng được logic ràng buộc thứ tự thời gian — mock ngẫu nhiên thuần tuý
(như ở retriever.py dummy của D) KHÔNG kiểm chứng được việc này vì similarity
không liên quan gì đến timestamp thật.
Chạy: python -m src.retrieval.demo_trake
"""
import random
from .event_sequence import run_trake, TrakeResult
from .temporal import TemporalConfig
from .submission import to_trake_submission, validate_submission_batch

# Ánh xạ từ khoá sự kiện -> timestamp "đúng" mong đợi trong video_JUMP (giả lập)
EVENT_KEYWORD_TIMESTAMP = {
    "chạy đà": 30.0,
    "giậm nhảy": 90.0,
    "bay qua xà": 150.0,
    "tiếp đất": 190.0,
}


def mock_search_fn_biased_to_one_video(query: str, top_k: int = 50) -> list[dict]:
    """
    Mock search_fn có ngữ nghĩa: với MỖI query (mô tả 1 sự kiện con), frame
    GẦN timestamp "đúng" của sự kiện đó (theo EVENT_KEYWORD_TIMESTAMP) sẽ có
    similarity cao nhất trong video_JUMP — mô phỏng đúng hành vi CLIP thật
    (frame càng khớp ngữ nghĩa càng similarity cao), khác hẳn dummy ngẫu
    nhiên hoàn toàn của D hiện tại (D-B1 chưa nối CLIP thật — xem phân tích
    trước đó về retriever.py).
    """
    query_lower = query.lower()
    target_ts = next(
        (ts for kw, ts in EVENT_KEYWORD_TIMESTAMP.items() if kw in query_lower),
        None,
    )

    results = []
    for i in range(20):
        ts = float(i * 10)
        if target_ts is not None:
            # similarity giảm dần theo khoảng cách tới target_ts -> đỉnh RÕ
            # RÀNG tại đúng thời điểm sự kiện, mô phỏng CLIP thật
            distance = abs(ts - target_ts)
            similarity = max(0.3, 0.95 - distance * 0.01)
        else:
            similarity = 0.6  # query gốc (Giai đoạn 1) — không thiên vị điểm nào
        results.append({
            "video_id": "video_JUMP", "frame_idx": i * 10, "timestamp": ts,
            "similarity": round(similarity, 3), "objects": [],
        })

    # Video nhiễu nền — similarity luôn thấp hơn hẳn video đúng
    for v in range(3):
        for i in range(10):
            results.append({
                "video_id": f"video_noise_{v}", "frame_idx": i, "timestamp": float(i),
                "similarity": round(random.uniform(0.2, 0.4), 3), "objects": [],
            })
    return sorted(results, key=lambda x: -x["similarity"])[:top_k]


def mock_decomposer(query: str) -> list[str]:
    """Giả lập LLM tách chuỗi 'nhảy cao' thành 4 sự kiện con cố định."""
    return [
        "khoảnh khắc chạy đà và bước qua vạch xuất phát",
        "khoảnh khắc giậm nhảy rời mặt đất",
        "khoảnh khắc bay qua xà ở điểm cao nhất",
        "khoảnh khắc tiếp đất",
    ]


def run_demo():
    query = "Vận động viên thực hiện cú nhảy cao: chạy đà, giậm nhảy, bay qua xà, tiếp đất"

    result: TrakeResult = run_trake(
        query=query,
        search_fn=mock_search_fn_biased_to_one_video,
        decomposer=mock_decomposer,
        temporal_config=TemporalConfig(threshold_method="percentile"),
    )

    print(f"Video được chọn (Giai đoạn 1): {result.video_id}")
    print(f"Số sự kiện con: {len(result.events)}")
    for e in result.events:
        print(f"  - '{e.event_description}' -> frame_idx={e.frame_idx}, "
              f"similarity={e.similarity}")

    print(f"\nHoàn chỉnh (mọi sự kiện đều định vị được): {result.is_complete}")
    print(f"frame_ids theo thứ tự: {result.frame_ids}")

    submission = to_trake_submission(result)
    print(f"\nSubmission format BTC: {submission}")

    err = validate_submission_batch([submission] if submission else [], "trake")
    print(f"Validate: {'OK' if err is None else err}")

    assert result.video_id == "video_JUMP", "Phải chọn đúng video có similarity cao nhất"
    assert result.is_complete, "Với mock data, tất cả 4 sự kiện phải định vị được"
    assert len(result.frame_ids) == 4

    timestamps = [e.timestamp for e in result.events]
    assert timestamps == sorted(timestamps), (
        f"Timestamp phải tăng dần theo đúng thứ tự sự kiện, nhận: {timestamps}"
    )

    assert submission is not None and submission["video_id"] == "video_JUMP"
    print("\n✅ demo_trake PASS (bao gồm kiểm tra thứ tự thời gian tăng dần)")


if __name__ == "__main__":
    run_demo()