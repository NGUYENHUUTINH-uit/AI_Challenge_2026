"""
event_sequence.py — Task TRAKE (Temporal Retrieval and Alignment of Key
Events), theo mục 1.3 đề thi chính thức AIC 2026.

BÀI TOÁN: Query mô tả 1 CHUỖI N sự kiện có thứ tự (vd. "nhảy cao": chạy đà ->
giậm nhảy -> bay qua xà -> tiếp đất). Cần: (1) tìm ĐÚNG 1 video chứa chuỗi
sự kiện này, (2) với video đó, định vị CHÍNH XÁC 1 khung hình (semantic
keyframe) cho MỖI sự kiện — đáp án thường hẹp dưới 10 frame (mục 1.3 đề thi).

THIẾT KẾ: tận dụng lại B4 (temporal.py) + search_fn (D-B1) thay vì viết
thuật toán mới hoàn toàn — 2 giai đoạn:

  Giai đoạn 1 (Retrieval — chọn video): TÁI SỬ DỤNG find_time_range() có sẵn.
  Segment có peak_similarity cao nhất trên toàn bộ kết quả search với query
  GỐC (mô tả chung của cả chuỗi) được coi là video đúng — logic giống hệt
  cách KIS chọn video, không cần thuật toán riêng.

  Giai đoạn 2 (Alignment — định vị từng sự kiện): TÁCH câu query thành N mô
  tả sự kiện con (bằng LLM — đây là bước BẮT BUỘC cần LLM, không có cách
  rule-based đủ tốt để tách câu mô tả hành động phức hợp thành các bước con
  chính xác). Với MỖI sự kiện con, search lại rồi LỌC kết quả chỉ giữ đúng
  video đã chọn ở Giai đoạn 1, lấy frame có similarity cao nhất làm khung
  hình đại diện cho sự kiện đó.

VÌ SAO LỌC PHÍA CLIENT (không yêu cầu retriever.py hỗ trợ tham số
video_id) — để KHÔNG phụ thuộc D phải sửa thêm search_keyframes() — đổi lại
kém hiệu quả hơn (tốn search dư thừa ở các video khác), chấp nhận được vì
TRAKE ít xảy ra hơn KIS/Q&A trong 1 phiên thi.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from .temporal import find_time_range, TemporalConfig


@dataclass
class EventFrame:
    """1 sự kiện con đã được định vị — hoặc chưa tìm thấy (frame_idx=None)."""
    event_description: str
    frame_idx: Optional[int] = None
    timestamp: Optional[float] = None
    similarity: Optional[float] = None


@dataclass
class TrakeResult:
    video_id: Optional[str]
    events: list[EventFrame] = field(default_factory=list)

    @property
    def frame_ids(self) -> list[Optional[int]]:
        """Danh sách frame_id theo ĐÚNG THỨ TỰ sự kiện — dùng để nộp bài (xem submission.py)."""
        return [e.frame_idx for e in self.events]

    @property
    def is_complete(self) -> bool:
        """True nếu MỌI sự kiện đều định vị được (không có None nào)."""
        return self.video_id is not None and all(e.frame_idx is not None for e in self.events)


def _naive_decompose(query: str) -> list[str]:
    """
    Fallback KHÔNG dùng LLM — tách câu theo dấu phẩy/số thứ tự đơn giản.
    CHẤT LƯỢNG THẤP, chỉ dùng khi chưa có LLM (test logic, không cần API key).
    Không nên dùng bản này khi thi thật — xem cảnh báo trong make_llm_decomposer().
    """
    import re
    parts = re.split(r",|;|\bvà\b|\bsau đó\b", query)
    parts = [p.strip(" .") for p in parts if p.strip(" .")]
    return parts if len(parts) > 1 else [query]


def make_llm_decomposer(client, model: str = "claude-haiku-4-5-20251001") -> Callable[[str], list[str]]:
    """
    Factory tạo decomposer dùng LLM thật — tách query mô tả 1 chuỗi hành động
    thành N mô tả sự kiện con, THEO ĐÚNG THỨ TỰ THỜI GIAN.

    Dùng Haiku (rẻ) vì đây là tác vụ phân tích cú pháp/tách câu, không cần
    hiểu hình ảnh — khác với qa_answer.py (cần Sonnet/vision).
    """
    import json

    def decomposer(query: str) -> list[str]:
        prompt = f"""Câu sau mô tả MỘT CHUỖI các sự kiện/hành động con diễn ra theo thứ tự thời gian:
"{query}"

Hãy tách thành danh sách các sự kiện con, MỖI sự kiện là 1 khoảnh khắc NGẮN
(không phải cả quá trình), theo ĐÚNG THỨ TỰ xảy ra. Trả về CHỈ 1 JSON array
các chuỗi (string), không thêm text nào khác. Ví dụ cho "cú nhảy cao":
["khoảnh khắc chạy đà và bước qua vạch xuất phát", "khoảnh khắc giậm nhảy rời mặt đất", "khoảnh khắc bay qua xà ở điểm cao nhất", "khoảnh khắc tiếp đất"]"""

        response = client.messages.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    return decomposer


def _select_best_video(
    query: str,
    search_fn: Callable[[str, int], list[dict]],
    top_k: int,
    temporal_config: Optional[TemporalConfig] = None,
) -> Optional[str]:
    """Giai đoạn 1: chọn 1 video duy nhất — tái sử dụng B4 (find_time_range)."""
    raw_results = search_fn(query, top_k)
    segments = find_time_range(raw_results, temporal_config or TemporalConfig())
    return segments[0]["video_id"] if segments else None


def _localize_event(
    event_query: str,
    target_video_id: str,
    search_fn: Callable[[str, int], list[dict]],
    top_k: int,
    min_timestamp: Optional[float] = None,
) -> EventFrame:
    """
    Giai đoạn 2: định vị 1 sự kiện con, CHỈ trong phạm vi target_video_id.

    min_timestamp: nếu có, CHỈ xét các frame có timestamp > min_timestamp —
    dùng để ép các sự kiện phải theo ĐÚNG THỨ TỰ THỜI GIAN (sự kiện thứ i+1
    phải xảy ra SAU sự kiện thứ i). Đây là ràng buộc BẮT BUỘC về mặt ngữ
    nghĩa của TRAKE (chuỗi sự kiện CÓ THỨ TỰ, xem mục 1.3 đề thi) — nếu không
    ép ràng buộc này, mỗi sự kiện được tìm ĐỘC LẬP có thể ra thứ tự thời gian
    lộn xộn (đã xảy ra trong lần chạy demo_trake.py đầu tiên: sự kiện cuối
    "tiếp đất" trả về frame_idx=0, tức ĐỨNG TRƯỚC "chạy đà" — vô nghĩa).

    Nếu áp ràng buộc mà KHÔNG có candidate nào (vd. LLM tách sai thứ tự sự
    kiện, hoặc search không đủ kết quả), fallback về KHÔNG ràng buộc — thà
    có 1 kết quả không hoàn hảo còn hơn bỏ trống hoàn toàn.
    """
    raw_results = search_fn(event_query, top_k)
    candidates = [r for r in raw_results if r["video_id"] == target_video_id]

    if min_timestamp is not None:
        constrained = [r for r in candidates if r["timestamp"] > min_timestamp]
        if constrained:
            candidates = constrained
        # else: fallback giữ nguyên candidates chưa lọc — xem docstring ở trên

    if not candidates:
        return EventFrame(event_description=event_query)

    best = max(candidates, key=lambda r: r["similarity"])
    return EventFrame(
        event_description=event_query,
        frame_idx=best.get("frame_idx"),
        timestamp=best.get("timestamp"),
        similarity=best.get("similarity"),
    )


def run_trake(
    query: str,
    search_fn: Callable[[str, int], list[dict]],
    decomposer: Optional[Callable[[str], list[str]]] = None,
    top_k_video: int = 50,
    top_k_event: int = 50,
    temporal_config: Optional[TemporalConfig] = None,
) -> TrakeResult:
    """
    Entry point chính cho TRAKE.

    decomposer: hàm tách query thành list sự kiện con — dùng make_llm_decomposer()
                khi có API key; None sẽ dùng _naive_decompose() (CHẤT LƯỢNG
                THẤP, chỉ để test logic không cần key).
    """
    decompose_fn = decomposer or _naive_decompose

    video_id = _select_best_video(query, search_fn, top_k_video, temporal_config)
    if video_id is None:
        return TrakeResult(video_id=None, events=[])

    event_queries = decompose_fn(query)

    events: list[EventFrame] = []
    last_timestamp: Optional[float] = None
    for eq in event_queries:
        ev = _localize_event(eq, video_id, search_fn, top_k_event, min_timestamp=last_timestamp)
        events.append(ev)
        if ev.timestamp is not None:
            last_timestamp = ev.timestamp  # ép sự kiện tiếp theo phải SAU thời điểm này

    return TrakeResult(video_id=video_id, events=events)