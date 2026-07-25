"""
agent.py — Task B5: Pipeline Agent hoàn chỉnh (Reasoning, Planning, Memory, ReAct)

Kiến trúc: ReAct rút gọn (Reason -> Act -> Observe), phù hợp bài toán
retrieval hơn ReAct đầy đủ nhiều bước, vì mỗi lượt chỉ cần 1 chu kỳ
quyết định -> tìm kiếm -> ghi nhớ, không cần lặp nhiều bước suy luận
liên tiếp trong cùng 1 lượt (điều đó sẽ tốn thời gian khi thi bấm giờ).

Thiết kế injectable (search_fn, time_range_fn, llm_resolver đều truyền từ
ngoài vào) để:
  1. Test được hoàn toàn KHÔNG cần API key và KHÔNG cần chờ retriever.py
     thật của D — dùng mock/stub.
  2. Khi D xong retriever.py thật, chỉ đổi 1 dòng ở nơi khởi tạo Agent,
     không sửa logic bên trong agent.py.
  3. Khi có API key, bật LLM resolver mà không phải sửa lại class.
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


class RetrievalAgent:
    def __init__(
        self,
        search_fn: Callable[[str, int], list[dict]],
        time_range_fn: Callable = find_time_range,
        llm_resolver: Optional[Callable[[str, str], dict]] = None,
        query_expander: Optional[Callable[[str], list[str]]] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        search_fn: hàm search_keyframes(query, top_k) -> list[dict] — D-B1.
        time_range_fn: hàm find_time_range(keyframes, config) -> list[dict] — B4.
        llm_resolver: xem planner.py — None nghĩa là chạy rule-based only (free).
        query_expander: hàm query -> list[str] biến thể — D-B2, tuỳ chọn ghép
                        vào Agent để mở rộng truy vấn trước khi search.
        """
        self.search_fn = search_fn
        self.time_range_fn = time_range_fn
        self.llm_resolver = llm_resolver
        self.query_expander = query_expander
        self.config = config or AgentConfig()

        self.memory = AgentMemory(
            short_term_limit=self.config.short_term_limit,
            summarize_keep_recent=self.config.summarize_keep_recent,
        )
        self._llm_cache: dict = {}

    def run(self, query: str) -> dict:
        """
        Vòng lặp ReAct 1 chu kỳ:
          REASON (plan_action) -> ACT (search + temporal) -> OBSERVE (memory)
        Trả về dict đầy đủ, kèm thông tin debug (action, timing) hữu ích khi
        tinh chỉnh hệ thống ở giai đoạn 2.
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
        t3 = time.time()

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
                  f"temporal={t3-t2:.3f}s total={t3-t0:.3f}s")

        return {
            "segments": segments,
            "action": plan.action,
            "resolved_query": plan.resolved_query,
            "used_llm": plan.used_llm,
            "reasoning": plan.reasoning,
            "latency_seconds": round(t3 - t0, 3),
        }