"""
test_b_role.py — Test tự động (assert) cho toàn bộ vai trò B, ghép với code
thật của D. Chạy: python -m src.retrieval.test_b_role
"""
from .agent import RetrievalAgent, AgentConfig
from .temporal import TemporalConfig
from .retriever import Retriever
from .adapter import make_search_fn, make_query_expander, MetadataLookup


def build_agent():
    retriever = Retriever(use_dummy=True)
    metadata = MetadataLookup("data/processed/metadata.json")
    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)
    config = AgentConfig(
        temporal_config=TemporalConfig(threshold_method="percentile", percentile=80),
        top_k=20,
    )
    return RetrievalAgent(search_fn=search_fn, query_expander=make_query_expander(), config=config)


def test_similarity_in_valid_range():
    agent = build_agent()
    result = agent.run("Một người đàn ông và chiếc xe đạp")
    for seg in result["segments"]:
        assert 0.0 <= seg["avg_similarity"] <= 1.0, f"similarity vượt phạm vi: {seg}"
        assert 0.0 <= seg["peak_similarity"] <= 1.0, f"similarity vượt phạm vi: {seg}"
    print(f"✅ test_similarity_in_valid_range PASS ({len(result['segments'])} segments)")


def test_video_id_not_mixed():
    agent = build_agent()
    result = agent.run("Một người đàn ông và chiếc xe đạp")
    for seg in result["segments"]:
        assert seg["start"] <= seg["end"], f"start > end: {seg}"
    print("✅ test_video_id_not_mixed PASS")


def test_kisc_refine_detected():
    agent = build_agent()
    agent.run("Một người đàn ông và chiếc xe đạp")
    result = agent.run("vậy lúc đó họ đang ở đâu?")
    assert result["action"] == "refine", f"Kỳ vọng 'refine', nhận '{result['action']}'"
    print("✅ test_kisc_refine_detected PASS")


def test_memory_records_correct_turns():
    agent = build_agent()
    agent.run("câu hỏi 1")
    agent.run("câu hỏi 2")
    assert len(agent.memory.history) == 2, f"Kỳ vọng 2 lượt, nhận {len(agent.memory.history)}"
    print("✅ test_memory_records_correct_turns PASS")


def test_agent_survives_search_error():
    """Đảm bảo Agent không sập nếu search_fn lỗi (fallback tầng B)."""
    def broken_search(query, top_k):
        raise RuntimeError("giả lập lỗi search")

    agent = RetrievalAgent(search_fn=broken_search)
    result = agent.run("câu hỏi bất kỳ")
    assert result["action"] == "error"
    assert result["segments"] == []
    print("✅ test_agent_survives_search_error PASS")


if __name__ == "__main__":
    test_similarity_in_valid_range()
    test_video_id_not_mixed()
    test_kisc_refine_detected()
    test_memory_records_correct_turns()
    test_agent_survives_search_error()
    print("\n🎉 Tất cả test PASS")