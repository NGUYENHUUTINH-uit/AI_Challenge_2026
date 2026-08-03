"""
benchmark_b_role.py — Đo hiệu suất (latency) từng bước của pipeline B4+B5.
Chạy: python -m src.retrieval.benchmark_b_role
"""
import time
import statistics
from .agent import RetrievalAgent, AgentConfig
from .temporal import TemporalConfig, find_time_range
from .retriever import Retriever
from .adapter import make_search_fn, make_query_expander, MetadataLookup


def benchmark_temporal_only(n_runs=50, n_frames=500):
    """Đo riêng B4 (find_time_range) — không phụ thuộc search/LLM."""
    import random
    frames = [
        {"video_id": f"v{i%20}", "timestamp": float(i % 100), "similarity": round(random.uniform(0.2, 0.95), 3)}
        for i in range(n_frames)
    ]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        find_time_range(frames, TemporalConfig(threshold_method="percentile"))
        times.append(time.perf_counter() - t0)
    print(f"[B4 - find_time_range] n_frames={n_frames}, n_runs={n_runs}")
    print(f"  mean={statistics.mean(times)*1000:.3f}ms  median={statistics.median(times)*1000:.3f}ms  "
          f"max={max(times)*1000:.3f}ms")


def benchmark_full_pipeline(n_runs=10):
    """Đo toàn bộ pipeline Agent (search + rerank + temporal), KHÔNG có LLM (free)."""
    retriever = Retriever(use_dummy=True)
    metadata = MetadataLookup("data/processed/metadata.json")
    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)
    config = AgentConfig(top_k=50)

    queries = [
        "Một người đàn ông và chiếc xe đạp",
        "vậy lúc đó họ đang ở đâu?",
        "còn chiếc xe màu gì?",
    ]

    total_times = []
    for _ in range(n_runs):
        agent = RetrievalAgent(search_fn=search_fn, query_expander=make_query_expander(), config=config)
        for q in queries:
            t0 = time.perf_counter()
            agent.run(q)
            total_times.append(time.perf_counter() - t0)

    print(f"\n[Full pipeline - agent.run()] {len(total_times)} lượt gọi (rule-based only, KHÔNG LLM)")
    print(f"  mean={statistics.mean(total_times)*1000:.2f}ms  median={statistics.median(total_times)*1000:.2f}ms  "
          f"max={max(total_times)*1000:.2f}ms  min={min(total_times)*1000:.2f}ms")

    # Đánh giá so với ràng buộc thi bấm giờ (mỗi lượt nên < 2-3s để không làm chậm thí sinh)
    p95 = sorted(total_times)[int(len(total_times) * 0.95)]
    verdict = "OK — an toàn cho thi bấm giờ" if p95 < 2.0 else "CẦN TỐI ƯU — vượt ngưỡng an toàn"
    print(f"  p95={p95*1000:.2f}ms -> {verdict}")


def benchmark_dedup_effectiveness():
    """Kiểm chứng cơ chế dedup (khử trùng frame giữa các query variant) có hiệu quả không."""
    retriever = Retriever(use_dummy=True)
    metadata = MetadataLookup("data/processed/metadata.json")
    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)

    query = "Một người đàn ông và chiếc xe đạp"
    variants = make_query_expander()(query)
    raw_count = 0
    for q in [query] + variants:
        raw_count += len(search_fn(q, top_k=50))

    agent = RetrievalAgent(search_fn=search_fn, query_expander=make_query_expander(),
                            config=AgentConfig(top_k=50))
    result = agent.run(query)
    n_segments = len(result["segments"])

    print(f"\n[Dedup check] Tổng frame thô từ {len(variants)+1} biến thể: {raw_count}")
    print(f"  Số segment sau khi B4 gom cụm (đã dedup): {n_segments}")
    print(f"  -> Với dummy random video_id, dedup thường không giảm nhiều (video_id ngẫu nhiên "
          f"trong 100 video, xác suất trùng thấp) — sẽ hiệu quả rõ hơn với data thật.")


if __name__ == "__main__":
    benchmark_temporal_only()
    benchmark_full_pipeline()
    benchmark_dedup_effectiveness()