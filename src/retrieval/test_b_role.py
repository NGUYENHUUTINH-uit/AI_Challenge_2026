"""
test_b_role.py — Bộ test tự động cho TOÀN BỘ vai trò B, khớp kiến trúc MỚI
(search_pipeline.py bọc Retriever thật của A). Chạy:
    python -m src.retrieval.test_b_role

MockRetriever bên dưới mô phỏng ĐÚNG schema thật retriever.py của A trả về
({"video_id","filename","timestamp","score","objects","image_path",
"scene_id"}) — KHÔNG dùng schema cũ (similarity/frame_idx) đã lỗi thời.
"""
import random

from .search_pipeline import SearchPipeline, PipelineConfig, to_kis_json, to_qa_json, to_trake_json
from .agent import RetrievalAgent, AgentConfig
from .agent_pool import AgentPool
from .qa_answer import make_mock_answer_fn


class MockRetriever:
    """Khớp đúng schema thật A trả về — xem retriever.py (Task A)."""

    def search_keyframes(self, query_text=None, query_image=None, top_k=20):
        out = []
        for i in range(top_k):
            out.append({
                "video_id": f"L01_V{i % 5:03d}",
                "filename": f"{i:04d}.jpg",
                "timestamp": float(i),
                "score": round(random.uniform(0.3, 0.95), 3),
                "objects": ["person", "bicycle"] if i % 2 == 0 else ["tree"],
                "image_path": f"data/processed/keyframes/L01_V{i % 5:03d}/{i:04d}.jpg",
                "scene_id": i // 5,
            })
        return out


class SemanticMockRetriever:
    """Similarity gắn đúng ngữ nghĩa theo timestamp — dùng để test TRAKE
    (thứ tự thời gian), KIS thuần ngẫu nhiên không kiểm chứng được việc này."""

    EVENT_TS = {"giậm nhảy": 90.0, "bay qua xà": 150.0, "tiếp đất": 190.0}

    def search_keyframes(self, query_text=None, query_image=None, top_k=20):
        target = self.EVENT_TS.get((query_text or "").strip())
        out = []
        for i in range(20):
            ts = float(i * 10)
            sim = max(0.3, 0.95 - abs(ts - target) * 0.01) if target else 0.6
            out.append({
                "video_id": "L_JUMP", "filename": f"{i*10:04d}.jpg", "timestamp": ts,
                "score": round(sim, 3), "objects": [],
                "image_path": f"kf/L_JUMP/{i*10:04d}.jpg", "scene_id": 0,
            })
        return sorted(out, key=lambda x: -x["score"])[:top_k]


def test_kis_basic():
    pipeline = SearchPipeline(retriever=MockRetriever())
    segments = pipeline.search_kis("người đi xe đạp", top_k=20)
    assert len(segments) > 0, "KIS phải trả về ít nhất 1 segment"
    for s in segments:
        assert 0.0 <= s["avg_score"], f"score âm bất thường: {s}"
        assert s["start"] <= s["end"]
    json_out = to_kis_json(segments)
    assert all("video_id" in j and "frame_id" in j for j in json_out)
    print(f"✅ test_kis_basic PASS ({len(segments)} segments)")


def test_object_rerank_boosts_score():
    pipeline = SearchPipeline(retriever=MockRetriever())
    raw = pipeline._raw_search("người đi xe đạp", top_k=20)
    reranked = pipeline.search_kis("người đi xe đạp", top_k=20)
    # Segment có matched_objects > 0 phải có avg_score được cộng thêm alpha
    boosted = [s for s in reranked if s.get("matched_objects", 0) > 0]
    assert len(boosted) > 0, "Phải có ít nhất 1 segment được boost object"
    print(f"✅ test_object_rerank_boosts_score PASS ({len(boosted)} segment boosted)")


def test_video_not_mixed():
    pipeline = SearchPipeline(retriever=MockRetriever())
    segments = pipeline.search_kis("bất kỳ", top_k=30)
    for s in segments:
        assert s["start"] <= s["end"], f"start > end trong cùng 1 video: {s}"
    print("✅ test_video_not_mixed PASS")


def test_qa_produces_answer():
    pipeline = SearchPipeline(retriever=MockRetriever(), answer_fn=make_mock_answer_fn("5 người"))
    segments = pipeline.search_qa("cảnh trao giải", "có bao nhiêu người?", top_k=20)
    assert segments[0].get("answer") == "5 người"
    qa_json = to_qa_json(segments)
    assert len(qa_json) >= 1 and qa_json[0]["answer"] == "5 người"
    print("✅ test_qa_produces_answer PASS")


def test_trake_temporal_order():
    pipeline = SearchPipeline(
        retriever=SemanticMockRetriever(),
        event_decomposer=lambda q: ["giậm nhảy", "bay qua xà", "tiếp đất"],
    )
    result = pipeline.search_trake("cú nhảy cao", top_k_video=20, top_k_event=20)
    assert result["video_id"] == "L_JUMP"
    frame_ids = [e["frame_id"] for e in result["events"]]
    assert all(f is not None for f in frame_ids), "Mọi sự kiện phải định vị được"
    assert frame_ids == sorted(frame_ids), f"frame_id phải tăng dần, nhận: {frame_ids}"
    submission = to_trake_json(result)
    assert submission is not None and submission["video_id"] == "L_JUMP"
    print(f"✅ test_trake_temporal_order PASS (frame_ids={frame_ids})")


def test_agent_kisc_memory():
    pipeline = SearchPipeline(retriever=MockRetriever())
    agent = RetrievalAgent(pipeline=pipeline, config=AgentConfig(top_k=20))
    agent.run("người mặc áo đỏ")
    agent.run("vậy lúc đó họ đang cầm gì?")
    assert len(agent.memory.history) == 2
    print("✅ test_agent_kisc_memory PASS")


def test_agent_survives_search_error():
    class BrokenRetriever:
        def search_keyframes(self, **kwargs):
            raise RuntimeError("giả lập lỗi search")

    pipeline = SearchPipeline(retriever=BrokenRetriever())
    agent = RetrievalAgent(pipeline=pipeline)
    result = agent.run("câu hỏi bất kỳ")
    assert result["action"] == "error"
    assert result["segments"] == []
    print("✅ test_agent_survives_search_error PASS")


def test_agent_pool_session_isolation():
    def factory():
        return RetrievalAgent(pipeline=SearchPipeline(retriever=MockRetriever()),
                               config=AgentConfig(top_k=10))

    pool = AgentPool(agent_factory=factory)
    pool.run("userA", "câu 1")
    pool.run("userB", "câu khác")
    pool.run("userA", "câu 2")
    assert len(pool.get_agent("userA").memory.history) == 2
    assert len(pool.get_agent("userB").memory.history) == 1
    print("✅ test_agent_pool_session_isolation PASS")


def test_query_expansion_timeout_fallback():
    from .search_pipeline import make_query_expander_llm

    class SlowClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                import time
                time.sleep(10)  # giả lập mạng nghẽn, lâu hơn timeout 3s

    expander = make_query_expander_llm(SlowClient(), timeout_seconds=0.5)
    variants = expander("câu hỏi bất kỳ")
    assert variants == [], "Phải trả về [] khi timeout, không được treo"
    print("✅ test_query_expansion_timeout_fallback PASS")


if __name__ == "__main__":
    test_kis_basic()
    test_object_rerank_boosts_score()
    test_video_not_mixed()
    test_qa_produces_answer()
    test_trake_temporal_order()
    test_agent_kisc_memory()
    test_agent_survives_search_error()
    test_agent_pool_session_isolation()
    test_query_expansion_timeout_fallback()
    print("\n🎉 Tất cả test PASS")