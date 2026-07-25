"""
planner.py — Reasoning + Planning cho RetrievalAgent (B5)

Kế hoạch ghi rõ B5 cần "Planning: Lập kế hoạch gọi tool thích hợp
(retrieval rập khuôn, mở rộng từ khóa, re-rank)". Đây là logic quyết định
riêng biệt, tách khỏi agent.py để dễ test và dễ thay đổi chiến lược (rule-based
→ LLM-based) mà không đụng vòng lặp ReAct chính.

CHIẾN LƯỢC CHI PHÍ (trả lời câu hỏi "miễn phí nhưng hiệu suất cao"):
  Không có cách nào miễn phí HOÀN TOÀN mà đạt độ chính xác ngang LLM khi cần
  hiểu ngữ nghĩa (vd. resolve đại từ "họ", "lúc đó" trỏ về ai/khi nào).
  Nhưng ta có thể giảm 70-90% lượt gọi LLM bằng cách:

  1. Rule-based (MIỄN PHÍ, <1ms) xử lý mọi câu KHÔNG có dấu hiệu tham chiếu
     ngữ cảnh — phần lớn câu hỏi trong 1 phiên KISC thực ra là câu hỏi mới
     (new_search), không cần LLM.
  2. CHỈ gọi LLM (Haiku — rẻ nhất, ~$0.25/1M input token) khi rule-based phát
     hiện dấu hiệu tham chiếu (đại từ, từ nối) VÀ đã có lịch sử hội thoại.
  3. Cache: nếu 2 câu hỏi giống hệt nhau xuất hiện lại (hay gặp khi thi thử
     nhiều lần với cùng bộ query test), không gọi lại LLM.

  → Với ngân sách sinh viên, cách này giúp cả mùa thi (hàng trăm câu hỏi thử)
    tốn chưa đến vài nghìn đồng tiền API, thay vì gọi LLM cho MỌI câu hỏi.
"""
from dataclasses import dataclass
from typing import Callable, Optional

# Từ khoá tham chiếu ngữ cảnh — dấu hiệu câu hỏi phụ thuộc vào lượt trước.
# Đặt ở module-level (không hardcode sâu trong hàm) để dễ mở rộng/tinh chỉnh.
REFERENCE_KEYWORDS = [
    "đó", "họ", "người này", "người đó", "nó", "lúc đó", "vậy",
    "còn", "sau đó", "trước đó", "tiếp theo", "cái đó", "chỗ đó",
]


@dataclass
class PlanResult:
    action: str                       # "new_search" | "refine" | "clarify"
    resolved_query: str               # câu query cuối cùng dùng để search
    reasoning: str                    # lý do, phục vụ debug/log
    used_llm: bool                    # có gọi LLM không (theo dõi chi phí)


def _has_reference_signal(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in REFERENCE_KEYWORDS)


def _rule_based_plan(query: str, memory) -> PlanResult:
    if not memory.history:
        return PlanResult("new_search", query, "Chưa có lịch sử -> câu hỏi mới", used_llm=False)

    if _has_reference_signal(query):
        # Phát hiện tham chiếu nhưng KHÔNG có LLM để resolve -> vẫn refine,
        # nhưng resolved_query giữ nguyên (chất lượng thấp hơn bản có LLM).
        return PlanResult(
            "refine", query,
            "Phát hiện từ tham chiếu ngữ cảnh, không có LLM nên giữ nguyên query",
            used_llm=False,
        )

    return PlanResult("new_search", query, "Không có dấu hiệu tham chiếu", used_llm=False)


def plan_action(
    query: str,
    memory,
    llm_resolver: Optional[Callable[[str, str], dict]] = None,
    cache: Optional[dict] = None,
) -> PlanResult:
    """
    Entry point chính.

    llm_resolver: hàm (query, context_string) -> {"action", "resolved_query", "reasoning"}
                  Chỉ được GỌI khi rule-based phát hiện tham chiếu ngữ cảnh.
                  Truyền None để chạy hoàn toàn miễn phí (rule-based only) —
                  hữu ích khi chưa có API key, vẫn test được toàn bộ pipeline.
    cache: dict tuỳ chọn để tránh gọi LLM lặp lại cho câu hỏi giống hệt đã xử lý.
    """
    base = _rule_based_plan(query, memory)

    # Chỉ leo thang lên LLM khi: có tham chiếu ngữ cảnh + có sẵn resolver + có lịch sử
    if base.action == "refine" and llm_resolver is not None:
        cache_key = f"{query}::{memory.get_context_string()}"
        if cache is not None and cache_key in cache:
            cached = cache[cache_key]
            return PlanResult(cached["action"], cached["resolved_query"],
                               cached["reasoning"] + " (cache)", used_llm=False)

        try:
            result = llm_resolver(query, memory.get_context_string())
            plan = PlanResult(
                result.get("action", "refine"),
                result.get("resolved_query", query),
                result.get("reasoning", "LLM resolve"),
                used_llm=True,
            )
            if cache is not None:
                cache[cache_key] = {
                    "action": plan.action,
                    "resolved_query": plan.resolved_query,
                    "reasoning": plan.reasoning,
                }
            return plan
        except Exception as e:
            # Fallback an toàn: nếu LLM lỗi/timeout, KHÔNG để cả Agent crash —
            # dùng lại kết quả rule-based (đúng tinh thần fallback của C5).
            return PlanResult(base.action, base.resolved_query,
                               f"LLM lỗi ({e}), dùng rule-based", used_llm=False)

    return base


def make_anthropic_resolver(client, model: str = "claude-haiku-4-5-20251001"):
    """
    Factory tạo llm_resolver dùng Anthropic API — chỉ tạo khi có API key.
    Model mặc định là Haiku (rẻ nhất, đủ tốt cho việc phân loại + resolve
    đại từ — không cần Sonnet cho tác vụ này).
    """
    import json

    def resolver(query: str, context: str) -> dict:
        prompt = f"""Bạn là bộ phân loại ý định cho hệ thống tìm kiếm video.
Lịch sử hội thoại:
{context if context else "(chưa có)"}

Câu hỏi mới: "{query}"

Trả về CHỈ 1 JSON object, không thêm text nào khác:
{{"action": "new_search" hoặc "refine", "resolved_query": "câu query đã thay đại từ tham chiếu bằng thông tin cụ thể từ lịch sử", "reasoning": "lý do ngắn gọn"}}"""

        response = client.messages.create(
            model=model,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    return resolver