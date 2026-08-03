"""
demo_integration.py — Test ghép NGUYÊN CODE THẬT của D (retriever.py,
query_expansion.py, rerank.py) với agent.py/temporal.py/memory.py/planner.py
của B, thông qua adapter.py.

Đặt USE_REAL_LLM = True sau khi đã chạy test_api_key.py thành công.
"""
import os
from dotenv import load_dotenv

from .agent import RetrievalAgent, AgentConfig
from .temporal import TemporalConfig
from .retriever import Retriever
from .adapter import make_search_fn, make_query_expander, MetadataLookup
from .planner import make_anthropic_resolver

load_dotenv()
# Mặc định False để bất kỳ ai trong nhóm (chưa có API key) cũng chạy được ngay
# không lỗi. Tự đổi thành True trên MÁY RIÊNG sau khi đã có key và chạy
# test_api_key.py thành công — không commit giá trị True lên nhánh chung.
USE_REAL_LLM = False


def run_integration_test():
    retriever = Retriever(use_dummy=True)  # sau này D đổi use_dummy=False khi có data thật
    metadata = MetadataLookup("data/processed/metadata.json")

    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)
    query_expander = make_query_expander()

    llm_resolver = None
    if USE_REAL_LLM:
        import anthropic
        client = anthropic.Anthropic()  # tự đọc ANTHROPIC_API_KEY từ .env
        llm_resolver = make_anthropic_resolver(client)

    config = AgentConfig(
        temporal_config=TemporalConfig(threshold_method="percentile", percentile=70),
        enable_timing_log=True,
        top_k=20,
    )
    agent = RetrievalAgent(
        search_fn=search_fn,
        query_expander=query_expander,  # bật D-B2
        llm_resolver=llm_resolver,       # bật LLM thật (Reasoning/Planning)
        config=config,
    )

    queries = [
        "Một người đàn ông và chiếc xe đạp",
        "vậy lúc đó họ đang ở đâu?",
    ]

    for q in queries:
        result = agent.run(q)
        print(f"\n=== Query: {q} ===")
        print(f"Action: {result['action']} | used_llm={result['used_llm']}")
        print(f"Resolved query: {result['resolved_query']}")
        print(f"Số segment tìm được: {len(result['segments'])}")
        for seg in result["segments"][:3]:
            print(f"  {seg}")

    print("\n--- Memory context ---")
    print(agent.memory.get_context_string())


if __name__ == "__main__":
    run_integration_test()