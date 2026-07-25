"""
memory.py — Bộ nhớ ngắn hạn/dài hạn cho RetrievalAgent (B5)

Tách riêng thành file memory giúp:
  1. Test độc lập logic nhớ/quên mà không cần chạy toàn bộ Agent.
  2. Dễ thay đổi chiến lược nén ngữ cảnh (summarization) mà không đụng agent.py.
  3. Đúng nguyên tắc single-responsibility — agent.py chỉ điều phối, không tự
     quản lý state hội thoại.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Turn:
    query: str
    action: str
    result_summary: str
    resolved_query: Optional[str] = None   # câu query đã thay đại từ tham chiếu
    focus_video_id: Optional[str] = None


class AgentMemory:
    """
    Chiến lược 2 tầng (miễn phí, không cần LLM để vận hành tầng 1):

    - Tầng ngắn hạn (short-term): giữ nguyên N lượt gần nhất, dùng trực tiếp,
      không mất thông tin, không tốn API call.
    - Tầng dài hạn (long-term): khi vượt ngưỡng, NÉN các lượt cũ thành 1 bản
      tóm tắt. Việc nén CẦN gọi LLM (không có cách nào nén ngữ nghĩa chính xác
      mà miễn phí) — nhưng vì chỉ nén khi vượt ngưỡng (không phải mỗi lượt),
      chi phí gần như không đáng kể trong suốt 1 phiên hội thoại.
    """

    def __init__(self, short_term_limit: int = 6, summarize_keep_recent: int = 3):
        self.short_term_limit = short_term_limit
        self.summarize_keep_recent = summarize_keep_recent
        self.history: list[Turn] = []
        self.long_term_summary: Optional[str] = None
        self.current_focus: Optional[str] = None  # video_id đang được "khoá" trong hội thoại

    def add_turn(
        self,
        query: str,
        action: str,
        result_summary: str,
        resolved_query: Optional[str] = None,
        focus_video_id: Optional[str] = None,
    ) -> None:
        self.history.append(Turn(query, action, result_summary, resolved_query, focus_video_id))
        if focus_video_id:
            self.current_focus = focus_video_id

    def needs_summarization(self) -> bool:
        return len(self.history) > self.short_term_limit

    def summarize_if_needed(self, summarizer: Optional[Callable[[str], str]] = None) -> None:
        """
        summarizer: hàm nhận text thô -> trả về text tóm tắt.
        Inject từ bên ngoài (thay vì gọi API trực tiếp trong class này) để:
          - Test được mà không cần API key (truyền summarizer giả lập).
          - Dễ đổi nhà cung cấp LLM (Claude/GPT/Gemini) mà không sửa memory.py.
        """
        if not self.needs_summarization():
            return

        to_compress = self.history[: -self.summarize_keep_recent]
        keep = self.history[-self.summarize_keep_recent :]

        raw_text = "\n".join(
            f"- Query: {t.query} | Hành động: {t.action} | Kết quả: {t.result_summary}"
            for t in to_compress
        )

        if summarizer is not None:
            new_summary = summarizer(raw_text)
        else:
            # Fallback miễn phí, không LLM: nối chuỗi rút gọn (kém hơn nhưng free)
            new_summary = f"[{len(to_compress)} lượt trước]: " + "; ".join(
                t.query for t in to_compress
            )

        self.long_term_summary = (
            f"{self.long_term_summary}\n{new_summary}" if self.long_term_summary else new_summary
        )
        self.history = keep

    def get_context_string(self) -> str:
        """Ghép long-term summary + short-term history thành prompt context."""
        parts = []
        if self.long_term_summary:
            parts.append(f"[Bối cảnh trước đó]\n{self.long_term_summary}")
        if self.history:
            recent = "\n".join(
                f"Lượt {i+1}: Query='{t.query}'"
                + (f" (ý thực: '{t.resolved_query}')" if t.resolved_query else "")
                + f" -> {t.result_summary}"
                for i, t in enumerate(self.history)
            )
            parts.append(f"[Gần đây]\n{recent}")
        return "\n\n".join(parts) if parts else ""

    def reset(self) -> None:
        self.history.clear()
        self.long_term_summary = None
        self.current_focus = None