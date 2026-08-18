"""
agent_pool.py — Quản lý nhiều phiên hội thoại (session) độc lập.

VÌ SAO CẦN FILE NÀY:
Một RetrievalAgent instance gắn liền 1 AgentMemory. Khi API phục vụ nhiều
người dùng đồng thời, dùng chung 1 Agent sẽ làm Memory của người này lẫn
sang người khác (state leakage) — sai cho bài toán KISC vốn cần ngữ cảnh
RIÊNG của từng cuộc hội thoại.

AgentPool tạo 1 Agent MỚI cho mỗi session_id chưa từng gặp, và tái sử dụng
đúng Agent đó cho các lượt sau của CÙNG session — tương tự cách các web
framework quản lý session (vd. Flask session, Django session), chỉ khác là
ở đây "state" được giữ là AgentMemory thay vì cookie/dữ liệu form.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agent import RetrievalAgent, AgentConfig


@dataclass
class _SessionEntry:
    agent: RetrievalAgent
    last_used: float = field(default_factory=time.time)


class AgentPool:
    def __init__(
        self,
        agent_factory: Callable[[], RetrievalAgent],
        session_ttl_seconds: int = 1800,  # 30 phút không hoạt động -> dọn dẹp
    ):
        """
        agent_factory: hàm KHÔNG tham số, trả về 1 RetrievalAgent MỚI mỗi lần gọi.
        Dùng factory (không truyền sẵn 1 instance) vì mỗi session cần Agent
        RIÊNG — nếu dùng chung 1 instance thì lỗi state leakage vẫn xảy ra.

        session_ttl_seconds: sau thời gian này không hoạt động, session bị dọn
        để tránh rò rỉ bộ nhớ (memory leak) khi chạy server dài hạn — quan trọng
        khi thi đấu chạy liên tục nhiều giờ.
        """
        self._agent_factory = agent_factory
        self._sessions: dict[str, _SessionEntry] = {}
        self._ttl = session_ttl_seconds

    def get_agent(self, session_id: str) -> RetrievalAgent:
        entry = self._sessions.get(session_id)
        if entry is None:
            entry = _SessionEntry(agent=self._agent_factory())
            self._sessions[session_id] = entry
        entry.last_used = time.time()
        return entry.agent

    def run(self, session_id: str, query: str) -> dict:
        """Entry point tiện dụng: C chỉ cần gọi pool.run(session_id, query)."""
        agent = self.get_agent(session_id)
        return agent.run(query)

    def cleanup_expired(self) -> int:
        """Dọn các session hết hạn. Gọi định kỳ (vd. mỗi phút) từ tầng C,
        hoặc gọi trước mỗi request nếu không muốn thêm scheduler riêng."""
        now = time.time()
        expired = [sid for sid, e in self._sessions.items() if now - e.last_used > self._ttl]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def reset_session(self, session_id: str) -> None:
        """Xoá hẳn 1 session — dùng khi người dùng bấm 'New Search' trên UI."""
        self._sessions.pop(session_id, None)

    def active_session_count(self) -> int:
        return len(self._sessions)