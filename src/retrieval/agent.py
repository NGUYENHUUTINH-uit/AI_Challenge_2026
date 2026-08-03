"""
agent.py — Task B5: Pipeline Agent hoàn chỉnh (Reasoning, Planning, Memory, ReAct)

Kiến trúc: ReAct rút gọn (Reason -> Act -> Observe), phù hợp bài toán
retrieval hơn ReAct đầy đủ nhiều bước, vì mỗi lượt chỉ cần 1 chu kỳ
quyết định -> tìm kiếm -> ghi nhớ, không cần lặp nhiều bước suy luận
liên tiếp trong cùng 1 lượt (điều đó sẽ tốn thời gian khi thi bấm giờ).

HỖ TRỢ 3 DẠNG TRUY VẤN THEO ĐỀ THI CHÍNH THỨC (AIC 2026):
  - Textual KIS: run(query) -> segments (video_id, frame range) — đã có từ đầu.
  - Q&A: run(query, question=...) -> segments KÈM "answer" cho segment top-1,
    dùng answer_fn (VQA) đọc ảnh khung hình đại diện để trả lời.
  - TRAKE: xem event_sequence.py (module riêng).
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .memory import AgentMemory
from .planner import plan_action, PlanResult
from .temporal import find_time_range, TemporalConfig


@dataclass
class AgentConfig:
    short_term_limit: int = 6
    summarize_keep_recent: int = 3
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
    enable_timing_log: bool = False
    top_k: int = 50
    max_segments_for_answer: int = 1  # số segment top đầu sẽ thử trả lời Q&A


def _attach_representative_frame(segments: list[dict], raw_results: list[dict]) -> None:
    """
    Mỗi segment từ B4 chỉ có (start, end) — một KHOẢNG thời gian, không phải
    1 khung hình cụ thể. Nhưng đề thi yêu cầu nộp CHÍNH XÁC 1 frame_id (KIS)
    hoặc cần 1 ảnh cụ thể để trả lời câu hỏi (Q&A). Hàm này tìm frame có
    similarity cao nhất TRONG khoảng [start, end] của mỗi segment, gắn thêm
    "frame_idx"/"image_path"/"representative_timestamp" — sửa segments TRỰC
    TIẾP (in-place), không đổi format trả về của find_time_range() (B4 vẫn
    độc lập, không cần biết về frame_idx/image_path).
    """
    by_video: dict[str, list[dict]] = {}
    for r in raw_results:
        by_video.setdefault(r["video_id"], []).append(r)

    for seg in segments:
        candidates = [
            r for r in by_video.get(seg["video_id"], [])
            if seg["start"] <= r["timestamp"] <= seg["end"]
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r["similarity"])
        seg["representative_timestamp"] = best["timestamp"]
        seg["frame_idx"] = best.get("frame_idx")
        seg["image_path"] = best.get("image_path")


class RetrievalAgent:
    def __init__(
        self,
        search_fn: Callable[[str, int], list[dict]],
        time_range_fn: Callable = find_time_range,
        llm_resolver: Optional[Callable[[str, str], dict]] = None,
        query_expander: Optional[Callable[[str], list[str]]] = None,
        answer_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        search_fn: hàm search_keyframes(query, top_k) -> list[dict] — D-B1.
        time_range_fn: hàm find_time_range(keyframes, config) -> list[dict] — B4.
        llm_resolver: xem planner.py — None nghĩa là chạy rule-based only (free).
        query_expander: hàm query -> list[str] biến thể — D-B2, tuỳ chọn.
        answer_fn: hàm (image_path, question) -> answer(str) — dùng cho dạng
                   truy vấn Q&A. Xem qa_answer.py. None nghĩa là KHÔNG hỗ trợ
                   Q&A (chỉ trả segments như KIS, bỏ qua "question" nếu có).
        """
        self.search_fn = search_fn
        self.time_range_fn = time_range_fn
        self.llm_resolver = llm_resolver
        self.query_expander = query_expander
        self.answer_fn = answer_fn
        self.config = config or AgentConfig()

        self.memory = AgentMemory(
            short_term_limit=self.config.short_term_limit,
            summarize_keep_recent=self.config.summarize_keep_recent,
        )
        self._llm_cache: dict = {}

    def run(self, query: str, question: Optional[str] = None) -> dict:
        """
        Vòng lặp ReAct 1 chu kỳ:
          REASON (plan_action) -> ACT (search + temporal [+ VQA nếu có question])
          -> OBSERVE (memory)

        question: nếu có giá trị (dạng truy vấn Q&A theo đề thi), sau khi tìm
        được segment top, gọi answer_fn() với ảnh khung hình đại diện để sinh
        "answer" — bỏ qua nếu answer_fn=None (chưa cấu hình VQA).
        """
        t0 = time.time()

        # 1. REASON
        plan: PlanResult = plan_action(
            query, self.memory, llm_resolver=self.llm_resolver, cache=self._llm_cache
        )
        t1 = time.time()

        # 2. ACT — search (có fallback an toàn nếu search lỗi)
        search_query = plan.resolved_query
        queries_to_try = [search_query]
        if self.query_expander is not None:
            try:
                queries_to_try += self.query_expander(search_query)
            except Exception:
                pass  # query expansion là tối ưu thêm, lỗi không được chặn pipeline chính

        try:
            # Dedup theo (video_id, timestamp): khi query_expander sinh nhiều biến
            # thể (D-B2), cùng 1 frame có thể xuất hiện lại ở kết quả của nhiều
            # biến thể khác nhau -- giữ lại bản có similarity cao nhất, tránh đếm
            # trùng khi B4 (find_time_range) gom cụm theo thời gian.
            best_by_key: dict = {}
            for q in queries_to_try:
                for item in self.search_fn(q, self.config.top_k):
                    key = (item["video_id"], item["timestamp"])
                    if key not in best_by_key or item["similarity"] > best_by_key[key]["similarity"]:
                        best_by_key[key] = item
            raw_results = list(best_by_key.values())
        except Exception as e:
            # Fallback tang B: neu search loi hoan toan, tra ve rong co kiem soat
            # thay vi de exception roi ra ngoai lam sap ca Agent.
            self.memory.add_turn(query, "error", f"search loi: {e}")
            return {"segments": [], "action": "error", "error": str(e)}
        t2 = time.time()

        # 3. Temporal localization (B4)
        segments = self.time_range_fn(raw_results, self.config.temporal_config)
        _attach_representative_frame(segments, raw_results)
        t3 = time.time()

        # 3b. Q&A: sinh answer cho (các) segment top nếu có question + answer_fn
        if question is not None and self.answer_fn is not None:
            for seg in segments[: self.config.max_segments_for_answer]:
                image_path = seg.get("image_path")
                seg["answer"] = self.answer_fn(image_path, question) if image_path else None
        t4 = time.time()

        # 4. OBSERVE — cập nhật Memory
        focus_video = segments[0]["video_id"] if segments else None
        summary = f"{len(segments)} đoạn liên quan" + (
            f", đoạn mạnh nhất video {focus_video}" if focus_video else ""
        )
        self.memory.add_turn(
            query=query,
            action=plan.action,
            result_summary=summary,
            resolved_query=plan.resolved_query if plan.resolved_query != query else None,
            focus_video_id=focus_video,
        )
        self.memory.summarize_if_needed()  # nén nếu vượt short_term_limit

        if self.config.enable_timing_log:
            print(f"[Timing] plan={t1-t0:.3f}s search={t2-t1:.3f}s "
                  f"temporal={t3-t2:.3f}s answer={t4-t3:.3f}s total={t4-t0:.3f}s")

        return {
            "segments": segments,
            "action": plan.action,
            "resolved_query": plan.resolved_query,
            "used_llm": plan.used_llm,
            "reasoning": plan.reasoning,
            "latency_seconds": round(t4 - t0, 3),
        }