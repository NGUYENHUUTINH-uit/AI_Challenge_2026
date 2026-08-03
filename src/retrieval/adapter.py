"""
adapter.py — Cầu nối giữa module thật của D (retriever.py, query_expansion.py,
rerank.py) và interface mà agent.py/temporal.py (B4/B5) kỳ vọng.

VÌ SAO CẦN FILE NÀY (không sửa trực tiếp agent.py hay retriever.py của D):
1. D dùng tên field "score"/"frame_idx", B dùng "similarity"/"timestamp" — đặt
   tên khác nhau vì 2 người code song song, độc lập theo đúng nguyên tắc Git
   (không cùng sửa 1 file). Đổi tên ở ĐÚNG 1 nơi (adapter) an toàn hơn sửa rải
   rác nhiều nơi trong temporal.py/agent.py đã test kỹ.
2. D trả về "frame_idx" (chỉ số khung hình), nhưng B4 (find_time_range) cần
   "timestamp" (giây) để gom cụm theo thời gian thực — cần tra cứu qua
   metadata.json (do A tạo) để đổi frame_idx -> timestamp.
3. retriever.py của D là 1 class (Retriever), phải khởi tạo instance rồi gọi
   method — còn agent.py của B kỳ vọng 1 hàm thuần search_fn(query, top_k).
   Adapter "bọc" instance thành hàm để inject vào RetrievalAgent.
"""
import json
from typing import Optional


class MetadataLookup:
    """
    Tra cứu timestamp thật từ metadata.json (do A tạo).
    metadata.json có dạng: {"{video_id}_frame_{idx}.jpg": {"video_id", "timestamp"}}
    """

    def __init__(self, metadata_path: str):
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._index: dict[tuple[str, int], float] = {}
        for filename, info in raw.items():
            try:
                idx = int(filename.rsplit("_frame_", 1)[1].split(".")[0])
            except (IndexError, ValueError):
                continue  # bỏ qua entry không đúng format, không để crash cả pipeline
            self._index[(info["video_id"], idx)] = info["timestamp"]

    def get_timestamp(self, video_id: str, frame_idx: int) -> Optional[float]:
        return self._index.get((video_id, frame_idx))


def make_search_fn(
    retriever,
    metadata: Optional[MetadataLookup] = None,
    fallback_fps: float = 1.0,
    apply_rerank: bool = True,
):
    """
    Bọc retriever.search_keyframes() (của D) thành hàm search_fn(query, top_k)
    đúng format temporal.py/agent.py kỳ vọng:
        {"video_id": str, "timestamp": float, "similarity": float, "objects": list}

    apply_rerank: có gọi rerank_by_objects() (D-B3) ngay trong bước search không.
                  Đặt True theo mặc định vì rerank cần trường "score"/"objects"
                  RAW (trước khi đổi tên) — làm ở đây tiện hơn làm sau trong agent.py.

    fallback_fps: dùng khi metadata KHÔNG có timestamp khớp video_id+frame_idx —
                  xảy ra ngay bây giờ vì retriever.py đang ở chế độ dummy, sinh
                  video_id/frame_idx ngẫu nhiên không khớp metadata.json mẫu.
                  Khi D chuyển use_dummy=False (data thật), metadata sẽ khớp đầy
                  đủ và nhánh fallback này sẽ không cần dùng tới nữa.
    """
    from .rerank import rerank_by_objects  # import cục bộ, tránh vòng lặp import

    def search_fn(query: str, top_k: int = 50) -> list[dict]:
        raw = retriever.search_keyframes(query_text=query, top_k=top_k)

        if apply_rerank:
            raw = rerank_by_objects(raw, query)

        out = []
        for r in raw:
            ts = metadata.get_timestamp(r["video_id"], r["frame_idx"]) if metadata else None
            if ts is None:
                ts = r["frame_idx"] / fallback_fps  # fallback có kiểm soát, không crash

            # Clamp về [0, 1] — rerank_by_objects() của D cộng dồn match_count*0.1
            # vào score gốc không giới hạn trần, nhưng B4 (temporal.py) diễn giải
            # "similarity" như điểm chuẩn hoá trong khoảng [0, 1].
            similarity = min(1.0, float(r["score"]))

            out.append({
                "video_id": r["video_id"],
                "timestamp": float(ts),
                "similarity": similarity,
                "frame_idx": r["frame_idx"],  # giữ lại — bắt buộc phải nộp frame_id
                                                # theo đề thi chính thức (KIS/Q&A/TRAKE),
                                                # không thể chỉ nộp timestamp.
                "objects": r.get("objects", []),
                "image_path": r.get("image_path"),
            })
        return out

    return search_fn


def make_query_expander():
    """
    Bọc expand_query() (D-B2) khớp interface query_expander mà agent.py kỳ vọng:
    query -> list[str] biến thể, KHÔNG bao gồm câu gốc (agent.py tự thêm câu
    gốc vào đầu danh sách truy vấn để tránh search trùng 2 lần).
    """
    from .query_expansion import expand_query

    def query_expander(query: str) -> list[str]:
        variants = expand_query(query)
        return [v for v in variants if v != query]

    return query_expander