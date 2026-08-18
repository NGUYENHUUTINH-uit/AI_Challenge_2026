"""
search_pipeline.py — Lớp điều phối B (Query Expansion, Object Re-ranking,
Temporal Grouping, Q&A, TRAKE) xây TRÊN NỀN Retriever thật của A
(retriever.py — Task A), KHÔNG sửa/đụng file đó.

VÌ SAO TÁCH FILE RIÊNG (không viết đè vào retriever.py):
A đã nhận retriever.py làm file riêng (comment "Task A (Search)"), có sẵn
FAISS+CLIP thật + dịch tiếng Việt tự động + đọc object 2 format. B không
cần (và không nên) làm lại các việc đó — chỉ cần LỚP ĐIỀU PHỐI phía trên,
dùng search_keyframes() của A làm nguyên liệu thô.

LỖ HỔNG CẦN A XÁC NHẬN GẤP (xem chi tiết cuối file): search_keyframes() của
A hiện KHÔNG trả "frame_id" (chỉ có "filename") — thiếu field bắt buộc để
nộp bài. Code dưới đây có suy đoán tạm thời từ filename, NHƯNG ĐỘ TIN CẬY
THẤP, cần A xác nhận đúng quy tắc đặt tên trước khi dùng để nộp bài thật.
"""
from __future__ import annotations

import json
import time
import concurrent.futures
import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable, Optional


@dataclass
class PipelineConfig:
    # Temporal grouping (B4)
    base_threshold: float = 0.55
    gap_tolerance: float = 2.0
    threshold_method: str = "percentile"
    percentile: float = 80.0
    mean_ratio: float = 0.9
    min_segment_frames: int = 1

    # Object re-ranking — dùng TRỰC TIẾP r["objects"] mà Retriever của A đã
    # load sẵn, KHÔNG tự đọc file JSON nữa (A đã làm việc đó).
    object_alpha: float = 0.15

    # Query Expansion
    query_expansion_timeout_seconds: float = 3.0
    query_expansion_max_variants: int = 4

    default_top_k: int = 100
    trake_top_k_video: int = 50
    trake_top_k_event: int = 50


# ═══════════════════════════════════════════════════════════════════════
# 1. QUERY EXPANSION — LLM, JSON có cấu trúc, timeout 3s bảo vệ hệ thống
# ═══════════════════════════════════════════════════════════════════════

def make_query_expander_llm(
    client,
    model: str = "claude-haiku-4-5-20251001",
    timeout_seconds: float = 3.0,
    max_variants: int = 4,
) -> Callable[[str], list[str]]:
    def _call_llm(query: str) -> list[str]:
        prompt = f"""Sinh {max_variants} biến thể chi tiết hơn cho câu truy vấn tìm kiếm video sau, \
giữ nguyên ý nghĩa cốt lõi, đa dạng cách diễn đạt.

Câu gốc: "{query}"

CHỈ trả về JSON array {max_variants} chuỗi, không thêm text nào khác."""
        response = client.messages.create(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = re.sub(r"^```json|```$", "", response.content[0].text.strip()).strip()
        variants = json.loads(text)
        if not isinstance(variants, list):
            raise ValueError("LLM không trả JSON array hợp lệ")
        return [str(v) for v in variants][:max_variants]

    def query_expander(query: str) -> list[str]:
        # KHÔNG dùng "with ThreadPoolExecutor()" — khi thoát khối with, Python
        # tự gọi shutdown(wait=True), BLOCK CHO ĐẾN KHI luồng chạy xong, dù
        # future.result() đã timeout từ trước! Đây là lỗi timeout "giả" —
        # phát hiện qua benchmark: cấu hình timeout=3s nhưng THỰC TẾ mất
        # đúng bằng thời gian API bị nghẽn (vd 10s) — MẤT HẲN Ý NGHĨA bảo vệ
        # hệ thống khi thi bấm giờ. Sửa: tạo executor thủ công, gọi
        # shutdown(wait=False) để trả về NGAY khi timeout, không chờ luồng
        # nền (luồng đó tự kết thúc sau, kết quả bị bỏ qua — chấp nhận được
        # vì đây chỉ là 1 lệnh gọi API mạng, không giữ tài nguyên quan trọng).
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call_llm, query)
        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)  # KHÔNG chờ — trả về ngay, đúng nghĩa timeout
            return []
        except Exception:
            executor.shutdown(wait=False)
            return []

    return query_expander


# ═══════════════════════════════════════════════════════════════════════
# 2. OBJECT RE-RANKING — dùng r["objects"] Retriever của A đã load sẵn
# ═══════════════════════════════════════════════════════════════════════

def extract_query_objects(query_text: str) -> list[str]:
    """Rule-based, MIỄN PHÍ — trích object nhắc tới trong câu query."""
    query_lower = query_text.lower()
    keyword_map = {
        "người": "person", "đàn ông": "man", "phụ nữ": "woman",
        "xe đạp": "bicycle", "xe hơi": "car", "ô tô": "car",
        "chó": "dog", "mèo": "cat", "cây": "tree",
        "xe máy": "motorcycle", "bàn": "table", "ghế": "chair",
    }
    return list({en for vi, en in keyword_map.items() if vi in query_lower})


def rerank_by_objects(raw_results: list[dict], query_text: str, cfg: PipelineConfig) -> list[dict]:
    """
    Cộng alpha vào score cho mỗi object khớp — object list LẤY TRỰC TIẾP từ
    r["objects"] (Retriever của A đã đọc file JSON sẵn), so khớp KHÔNG phân
    biệt hoa/thường vì BTC/YOLOv8m có thể trả nhãn khác cách viết hoa.
    """
    query_objects = extract_query_objects(query_text)
    if not query_objects:
        return raw_results

    out = []
    for r in raw_results:
        frame_objects_lower = {str(o).lower() for o in r.get("objects", [])}
        matched = sum(1 for qo in query_objects if qo in frame_objects_lower)
        new_r = dict(r)
        new_r["score"] = r["score"] + matched * cfg.object_alpha
        new_r["matched_objects"] = matched
        out.append(new_r)

    out.sort(key=lambda x: -x["score"])
    return out


# ═══════════════════════════════════════════════════════════════════════
# 3. TEMPORAL GROUPING (B4) — dùng "timestamp" A trả về, fallback filename
#    nếu timestamp None (đề phòng video thiếu media-info/fps)
# ═══════════════════════════════════════════════════════════════════════

def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _dynamic_threshold(scores: list[float], cfg: PipelineConfig) -> float:
    if not scores:
        return cfg.base_threshold
    if cfg.threshold_method == "mean":
        return max(cfg.base_threshold, (sum(scores) / len(scores)) * cfg.mean_ratio)
    if cfg.threshold_method == "percentile":
        return max(cfg.base_threshold, _percentile(scores, cfg.percentile))
    if cfg.threshold_method == "gap_based":
        s = sorted(scores, reverse=True)
        if len(s) < 2:
            return cfg.base_threshold
        gaps = [(s[i] - s[i + 1], i) for i in range(len(s) - 1)]
        return max(cfg.base_threshold, s[max(gaps, key=lambda x: x[0])[1]])
    raise ValueError(f"threshold_method không hợp lệ: {cfg.threshold_method}")


def group_temporal(raw_results: list[dict], cfg: PipelineConfig) -> list[dict]:
    """Gom cụm theo timestamp; nếu thiếu timestamp (None), dùng frame_id làm
    proxy thứ tự — KHÔNG loại bỏ frame chỉ vì thiếu timestamp."""
    enriched = []
    for r in raw_results:
        ts = r.get("timestamp")
        fid = r.get("frame_id")
        if ts is None:
            ts = float(fid) if fid is not None else 0.0
        enriched.append({**r, "_ts": ts, "_frame_id": fid})

    if not enriched:
        return []

    sorted_frames = sorted(enriched, key=lambda x: (x["video_id"], x["_ts"]))
    all_segments = []

    for video_id, group in groupby(sorted_frames, key=lambda x: x["video_id"]):
        frames = list(group)
        scores = [f["score"] for f in frames]
        threshold = _dynamic_threshold(scores, cfg)
        relevant = [f for f in frames if f["score"] >= threshold]
        if not relevant:
            continue

        segments, current = [], [relevant[0]]
        for f in relevant[1:]:
            if f["_ts"] - current[-1]["_ts"] <= cfg.gap_tolerance:
                current.append(f)
            else:
                segments.append(current)
                current = [f]
        segments.append(current)

        for seg in segments:
            if len(seg) < cfg.min_segment_frames:
                continue
            peak = max(seg, key=lambda x: x["score"])
            all_segments.append({
                "video_id": video_id,
                "start": seg[0]["_ts"],
                "end": seg[-1]["_ts"],
                "avg_score": round(sum(x["score"] for x in seg) / len(seg), 4),
                "peak_score": round(peak["score"], 4),
                "n_frames": len(seg),
                "frame_id": peak["_frame_id"],       # ⚠️ suy đoán tạm, xem mục 3
                "filename": peak.get("filename"),
                "image_path": peak.get("image_path"),  # A đã build sẵn đường dẫn thật
                "matched_objects": peak.get("matched_objects", 0),
            })

    return sorted(all_segments, key=lambda x: -x["peak_score"])


# ═══════════════════════════════════════════════════════════════════════
# 4. PIPELINE CHÍNH — gọi Retriever thật của A, thêm các lớp B ở trên
# ═══════════════════════════════════════════════════════════════════════

class SearchPipeline:
    def __init__(
        self,
        retriever,  # instance Retriever thật của A (retriever.py) — INJECT,
                    # không tự tạo bên trong, để test được với mock.
        query_expander: Optional[Callable[[str], list[str]]] = None,
        answer_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        event_decomposer: Optional[Callable[[str], list[str]]] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.retriever = retriever
        self.query_expander = query_expander
        self.answer_fn = answer_fn
        self.event_decomposer = event_decomposer
        self.config = config or PipelineConfig()

    def _raw_search(self, query_text: str, top_k: int) -> list[dict]:
        """Dịch tiếng Việt ĐÃ được Retriever của A tự lo bên trong
        search_keyframes() — B KHÔNG dịch lại, chỉ mở rộng query rồi gọi."""
        queries = [query_text]
        if self.query_expander is not None:
            queries += self.query_expander(query_text)

        best: dict[tuple, dict] = {}
        for q in queries:
            results = self.retriever.search_keyframes(query_text=q, top_k=top_k)
            for r in results:
                key = (r["video_id"], r.get("filename"))
                if key not in best or r["score"] > best[key]["score"]:
                    best[key] = r
        return list(best.values())

    def search_kis(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        top_k = top_k or self.config.default_top_k
        raw = self._raw_search(query_text, top_k)
        raw = rerank_by_objects(raw, query_text, self.config)
        return group_temporal(raw, self.config)

    def search_qa(self, query_text: str, question: str, top_k: Optional[int] = None) -> list[dict]:
        segments = self.search_kis(query_text, top_k)
        if segments and self.answer_fn is not None:
            top = segments[0]
            image_path = top.get("image_path")  # A đã build sẵn đường dẫn ảnh THẬT
            top["answer"] = self.answer_fn(image_path, question) if image_path else None
        return segments

    def search_trake(
        self, query_text: str,
        top_k_video: Optional[int] = None, top_k_event: Optional[int] = None,
    ) -> dict:
        top_k_video = top_k_video or self.config.trake_top_k_video
        top_k_event = top_k_event or self.config.trake_top_k_event

        segments = self.search_kis(query_text, top_k_video)
        if not segments:
            return {"video_id": None, "events": []}
        video_id = segments[0]["video_id"]

        decompose = self.event_decomposer or (lambda q: [q])
        event_queries = decompose(query_text)

        events, last_ts = [], None
        for eq in event_queries:
            raw = self._raw_search(eq, top_k_event)
            candidates = [r for r in raw if r["video_id"] == video_id]
            for c in candidates:
                c["_ts"] = c.get("timestamp") or float((c.get("filename")) or 0)
            if last_ts is not None:
                constrained = [c for c in candidates if c["_ts"] > last_ts]
                if constrained:
                    candidates = constrained
            if not candidates:
                events.append({"event": eq, "frame_id": None, "score": None})
                continue
            best = max(candidates, key=lambda c: c["score"])
            events.append({
                "event": eq, "frame_id": best.get("filename"),
                "score": best["score"],
            })
            last_ts = best["_ts"]

        return {"video_id": video_id, "events": events}


# ═══════════════════════════════════════════════════════════════════════
# CHUYỂN SANG FORMAT NỘP BÀI
# ═══════════════════════════════════════════════════════════════════════

def to_kis_json(segments: list[dict], max_answers: int = 100) -> list[dict]:
    return [{"video_id": s["video_id"], "frame_id": s["frame_id"]}
            for s in segments if s.get("frame_id") is not None][:max_answers]


def to_qa_json(segments: list[dict], max_answers: int = 100) -> list[dict]:
    return [{"video_id": s["video_id"], "frame_id": s["frame_id"], "answer": s["answer"]}
            for s in segments if s.get("frame_id") is not None and s.get("answer")][:max_answers]


def to_trake_json(trake_result: dict) -> Optional[dict]:
    if trake_result["video_id"] is None:
        return None
    frame_ids = [e["frame_id"] for e in trake_result["events"] if e["frame_id"] is not None]
    if not frame_ids:
        return None
    return {"video_id": trake_result["video_id"], "frame_ids": frame_ids}


if __name__ == "__main__":
    # Test với MOCK Retriever khớp đúng schema thật của A (không cần
    # FAISS/CLIP thật) — xem test_search_pipeline.py để test đầy đủ hơn.
    class MockRetriever:
        def search_keyframes(self, query_text=None, query_image=None, top_k=20):
            import random
            return [{
                "video_id": f"L01_V{i%3:03d}", "filename": f"{i:04d}.jpg",
                "timestamp": float(i), "score": round(random.uniform(0.3, 0.95), 3),
                "objects": ["person", "bicycle"] if i % 2 == 0 else ["tree"],
                "image_path": f"data/processed/keyframes/L01_V{i%3:03d}/{i:04d}.jpg",
                "scene_id": i // 5,
            } for i in range(top_k)]

    pipeline = SearchPipeline(retriever=MockRetriever())
    segments = pipeline.search_kis("người đi xe đạp", top_k=20)
    for s in segments[:3]:
        print(s)