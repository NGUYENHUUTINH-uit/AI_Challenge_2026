"""
agent.py — Task B5: Reasoning + Planning + Memory cho hội thoại nhiều lượt
(KISC).
"""
import time
from dataclasses import dataclass
from typing import Optional

from .memory import AgentMemory
from .planner import plan_action, PlanResult
from .search_pipeline import SearchPipeline


@dataclass
class AgentConfig:
    short_term_limit: int = 6
    summarize_keep_recent: int = 3
    enable_timing_log: bool = False
    top_k: int = 100


class RetrievalAgent:
    """
    Điều phối hội thoại nhiều lượt (KISC) cho dạng KIS/Q&A. TRAKE thường là
    truy vấn 1 lượt (không hội thoại theo đề thi), nên gọi thẳng
    pipeline.search_trake() từ tầng C, KHÔNG đi qua Agent này.
    """

    def __init__(
        self,
        pipeline: SearchPipeline,
        llm_resolver: Optional[callable] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        pipeline: instance SearchPipeline (search_pipeline.py) đã khởi tạo
                  sẵn, bọc quanh Retriever thật của A — Agent chỉ ĐIỀU PHỐI,
                  không tự search.
        llm_resolver: xem planner.py — resolve đại từ tham chiếu ngữ cảnh
                      ("lúc đó", "họ"...) bằng LLM. None = rule-based (free).
        """
        self.pipeline = pipeline
        self.llm_resolver = llm_resolver
        self.config = config or AgentConfig()

        self.memory = AgentMemory(
            short_term_limit=self.config.short_term_limit,
            summarize_keep_recent=self.config.summarize_keep_recent,
        )
        self._llm_cache: dict = {}

    def run(self, query: str, question: Optional[str] = None) -> dict:
        """
        1 chu kỳ ReAct rút gọn: REASON (plan_action) -> ACT (giao cho
        search_pipeline.py) -> OBSERVE (memory).

        question: có giá trị -> dạng Q&A (pipeline.search_qa()).
                  None -> dạng KIS (pipeline.search_kis()).
        """
        t0 = time.time()

        plan: PlanResult = plan_action(
            query, self.memory, llm_resolver=self.llm_resolver, cache=self._llm_cache
        )
        t1 = time.time()

        try:
            if question is not None:
                segments = self.pipeline.search_qa(
                    plan.resolved_query, question, top_k=self.config.top_k
                )
            else:
                segments = self.pipeline.search_kis(
                    plan.resolved_query, top_k=self.config.top_k
                )
        except Exception as e:
            self.memory.add_turn(query, "error", f"search loi: {e}")
            return {"segments": [], "action": "error", "error": str(e)}
        t2 = time.time()

        focus_video = segments[0]["video_id"] if segments else None
        summary = f"{len(segments)} doan lien quan" + (
            f", manh nhat video {focus_video}" if focus_video else ""
        )
        self.memory.add_turn(
            query=query, action=plan.action, result_summary=summary,
            resolved_query=plan.resolved_query if plan.resolved_query != query else None,
            focus_video_id=focus_video,
        )
        self.memory.summarize_if_needed()

        if self.config.enable_timing_log:
            print(f"[Timing] plan={t1-t0:.3f}s search+group={t2-t1:.3f}s total={t2-t0:.3f}s")

        return {
            "segments": segments,
            "action": plan.action,
            "resolved_query": plan.resolved_query,
            "used_llm": plan.used_llm,
            "reasoning": plan.reasoning,
            "latency_seconds": round(t2 - t0, 3),
        }