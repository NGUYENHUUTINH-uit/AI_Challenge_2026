"""
benchmark_b_role.py — Đo hiệu suất pipeline mới (search_pipeline.py + agent.py).
Chạy: python -m src.retrieval.benchmark_b_role
"""
import time
import statistics

from .search_pipeline import SearchPipeline
from .agent import RetrievalAgent, AgentConfig
from .test_b_role import MockRetriever


def benchmark_search_kis(n_runs: int = 30, top_k: int = 100):
    pipeline = SearchPipeline(retriever=MockRetriever())
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pipeline.search_kis("người đi xe đạp trên đường", top_k=top_k)
        times.append(time.perf_counter() - t0)

    print(f"[search_kis] n_runs={n_runs}, top_k={top_k}")
    print(f"  mean={statistics.mean(times)*1000:.2f}ms  median={statistics.median(times)*1000:.2f}ms  "
          f"max={max(times)*1000:.2f}ms")


def benchmark_full_agent(n_runs: int = 10):
    pipeline = SearchPipeline(retriever=MockRetriever())
    queries = [
        "người mặc áo đỏ đi qua cổng",
        "vậy lúc đó họ đang cầm gì?",
        "còn chiếc xe màu gì?",
    ]

    total_times = []
    for _ in range(n_runs):
        agent = RetrievalAgent(pipeline=pipeline, config=AgentConfig(top_k=50))
        for q in queries:
            t0 = time.perf_counter()
            agent.run(q)
            total_times.append(time.perf_counter() - t0)

    print(f"\n[agent.run() - rule-based, KHÔNG LLM] {len(total_times)} lượt gọi")
    print(f"  mean={statistics.mean(total_times)*1000:.2f}ms  "
          f"median={statistics.median(total_times)*1000:.2f}ms  max={max(total_times)*1000:.2f}ms")

    p95 = sorted(total_times)[int(len(total_times) * 0.95)]
    verdict = "OK — an toàn cho thi bấm giờ" if p95 < 2.0 else "CẦN TỐI ƯU — vượt ngưỡng an toàn"
    print(f"  p95={p95*1000:.2f}ms -> {verdict}")


def benchmark_query_expansion_timeout_cost():
    """Đo chi phí THỰC TẾ khi timeout xảy ra — đảm bảo không làm chậm quá
    đáng kể tổng thời gian phản hồi (quan trọng khi thi bấm giờ)."""
    from .search_pipeline import make_query_expander_llm

    class SlowClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                time.sleep(10)

    expander = make_query_expander_llm(SlowClient(), timeout_seconds=3.0)
    t0 = time.perf_counter()
    expander("câu hỏi bất kỳ")
    elapsed = time.perf_counter() - t0
    print(f"\n[Query Expansion timeout] Thời gian thực tế khi mạng nghẽn: {elapsed:.2f}s "
          f"(kỳ vọng ~3.0s, đúng bằng timeout_seconds cấu hình)")


if __name__ == "__main__":
    benchmark_search_kis()
    benchmark_full_agent()
    benchmark_query_expansion_timeout_cost()