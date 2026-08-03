"""
demo_qa.py — Test luồng Q&A: KHÔNG cần API key/ảnh thật, dùng mock answer_fn
để xác nhận logic (chọn đúng frame đại diện, gọi đúng answer_fn, gắn đúng
field "answer" vào segment) chạy đúng trước khi có key/data thật.
"""
from .agent import RetrievalAgent, AgentConfig
from .temporal import TemporalConfig
from .retriever import Retriever
from .adapter import make_search_fn, make_query_expander, MetadataLookup
from .qa_answer import make_mock_answer_fn


def run_qa_demo():
    retriever = Retriever(use_dummy=True)
    metadata = MetadataLookup("data/processed/metadata.json")
    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)

    agent = RetrievalAgent(
        search_fn=search_fn,
        query_expander=make_query_expander(),
        answer_fn=make_mock_answer_fn("có 5 người"),  # giả lập VQA trả lời cố định
        config=AgentConfig(top_k=30, max_segments_for_answer=1),
    )

    # Mô phỏng đúng format Q&A trong đề thi: query (mô tả) + question (câu hỏi)
    query = "cảnh lễ trao giải thưởng âm nhạc"
    question = "có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?"

    result = agent.run(query, question=question)

    print(f"Action: {result['action']}")
    print(f"Số segment: {len(result['segments'])}")
    top = result["segments"][0] if result["segments"] else None
    if top:
        print(f"Segment top-1: video_id={top['video_id']}, "
              f"frame_idx={top.get('frame_idx')}, "
              f"representative_timestamp={top.get('representative_timestamp')}")
        print(f"Answer (mock VQA): {top.get('answer')}")
        assert "answer" in top, "Segment top-1 phải có field 'answer' khi có question"
        assert top["answer"] == "có 5 người", "answer phải khớp mock_answer_fn"
    print("\n✅ demo_qa PASS — luồng Q&A hoạt động đúng logic (chưa test VQA thật)")


if __name__ == "__main__":
    run_qa_demo()