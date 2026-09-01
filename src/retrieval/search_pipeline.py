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
    base_threshold: float = 0.25   # Hạ từ 0.40 → 0.25 để không bỏ sót frame liên quan
    gap_tolerance: float = 2.0
    threshold_method: str = "percentile"
    percentile: float = 75.0
    mean_ratio: float = 0.85
    min_segment_frames: int = 1

    # Multi-Modal Weights (Fusion)
    object_alpha: float = 0.18        # Điểm thưởng cho mỗi object khớp
    ocr_alpha: float = 0.35           # Điểm thưởng cực mạnh khi khớp chữ/số trong OCR
    max_object_bonus: float = 0.40    # Tránh tràn điểm

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
    model: str = "gemini-2.5-flash",
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
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call_llm, query)
        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False)
            return []
        except Exception:
            executor.shutdown(wait=False)
            return []

    return query_expander


# ═══════════════════════════════════════════════════════════════════════
# 2. OBJECT & OCR RE-RANKING — TỪ ĐIỂN MỞ RỘNG + FUSION SCORE
# ═══════════════════════════════════════════════════════════════════════

# Từ điển ánh xạ tiếng Việt -> COCO & OpenImages classes
VIETNAMESE_OBJECT_MAP = {
    # Con người
    "người": "person", "đàn ông": "person", "nam": "person", "phụ nữ": "person",
    "nữ": "person", "cô gái": "person", "chàng trai": "person", "trẻ em": "person",
    "em bé": "person", "học sinh": "person", "sinh viên": "person",
    "cảnh sát": "person", "bác sĩ": "person",
    "giáo viên": "person", "thầy giáo": "person", "cô giáo": "person",
    "giáo viên nam": "person", "giáo viên nữ": "person",
    "người dẫn chương trình": "person", "phóng viên": "person",
    "diễn giả": "person", "thuyết trình viên": "person",

    # Phương tiện giao thông
    "xe máy": "motorcycle", "mô tô": "motorcycle", "xe gắn máy": "motorcycle",
    "xe đạp": "bicycle", "ô tô": "car", "xe hơi": "car", "xe con": "car",
    "xe buýt": "bus", "xe bus": "bus", "xe tải": "truck", "xe bán tải": "truck",
    "xe cứu thương": "car", "xe taxi": "car", "máy bay": "airplane",
    "trực thăng": "airplane", "tàu hỏa": "train", "xe lửa": "train",
    "thuyền": "boat", "tàu thủy": "boat", "cano": "boat", "ván trượt": "skateboard",

    # Động vật
    "chó": "dog", "cún": "dog", "mèo": "cat", "chim": "bird", "bồ câu": "bird",
    "ngựa": "horse", "bò": "cow", "trâu": "cow", "cừu": "sheep", "dê": "sheep",
    "voi": "elephant", "gấu": "bear", "khỉ": "monkey", "hươu": "deer",

    # Đồ đạc & Phụ kiện
    "ba lô": "backpack", "balo": "backpack", "cặp": "backpack", "túi xách": "handbag",
    "túi": "handbag", "vali": "suitcase", "ô": "umbrella", "dù": "umbrella",
    "mũ": "hat", "nón": "hat", "kính": "glasses", "kính mắt": "glasses",
    "kính đeo mắt": "glasses", "cà vạt": "tie",
    "điện thoại": "cell phone", "điện thoại di động": "cell phone",
    "laptop": "laptop", "máy tính": "laptop", "máy tính xách tay": "laptop",
    "máy tính bảng": "laptop",

    # Vật dụng trong nhà
    "bàn": "dining table", "ghế": "chair", "ghế dài": "couch", "sofa": "couch",
    "ghế sofa": "couch", "đi văng": "couch",
    "giường": "bed", "tivi": "tv", "tủ lạnh": "refrigerator", "đồng hồ": "clock",
    "cốc": "cup", "cốc nước": "cup", "ly": "cup", "ly nước": "cup",
    "chai": "bottle", "bình": "bottle", "bát": "bowl",
    "đĩa": "bowl", "dao": "knife",
    "bảng": "whiteboard", "bảng trắng": "whiteboard", "bảng đen": "blackboard",
    "màn hình": "tv", "máy chiếu": "tv",

    # Ngoài trời & Giao thông
    "đèn giao thông": "traffic light", "đèn đỏ": "traffic light",
    "biển báo": "stop sign", "cây": "tree", "cây cối": "tree",
    "ghế đá": "bench", "thùng rác": "waste container",

    # Đồ ăn
    "bánh mì": "sandwich", "chuối": "banana", "táo": "apple", "pizza": "pizza",
    "bánh ngọt": "cake",

    # Nhạc cụ & Sân khấu / Trường học
    "trống": "drum", "bộ trống": "drum", "trống cơ": "drum", "trống điện": "drum",
    "đàn": "piano", "piano": "piano", "đàn piano": "piano", "đàn organ": "piano",
    "đàn guitar": "guitar", "guitar": "guitar", "ghi ta": "guitar", "đàn violon": "violin",
    "micro": "microphone", "sân khấu": "stage", "khăn đỏ": "tie", "đồng phục": "person",
}



def extract_query_objects(query_text: str) -> list[str]:
    """Trích xuất danh sách object class (tiếng Anh) từ câu query tiếng Việt."""
    if isinstance(query_text, list):
        query_text = " ".join([str(q) for q in query_text])
    
    if not isinstance(query_text, str):
        return []
    
    query_lower = query_text.lower()
    matched_classes = set()

    # Sắp xếp từ khóa dài trước để ưu tiên cụm từ (vd: "đèn giao thông" trước "đèn")
    for vi_word in sorted(VIETNAMESE_OBJECT_MAP.keys(), key=lambda x: -len(x)):
        if f" {vi_word} " in f" {query_lower} " or vi_word in query_lower:
            matched_classes.add(VIETNAMESE_OBJECT_MAP[vi_word])

    return list(matched_classes)


def rerank_by_objects(
    raw_results: list[dict],
    query_text: str,
    cfg: Optional[PipelineConfig] = None,
    extra_objects: Optional[list[str]] = None,
    extra_ocr_keywords: Optional[list[str]] = None,
) -> list[dict]:
    """
    Multi-modal Score Fusion:
    Score = Score_CLIP + (matched_objects * object_alpha) + (matched_ocr * ocr_alpha)
    Hỗ trợ kết hợp cả từ điển tĩnh tiếng Việt và đối tượng/OCR do LLM trích xuất tự động.
    """
    if cfg is None:
        cfg = PipelineConfig()

    if not raw_results:
        return []

    # 1. Trích xuất object và OCR keywords từ query (kết hợp Heuristic + LLM)
    from src.indexing.ocr_indexer import extract_potential_ocr_keywords, get_ocr_indexer
    query_objects = set(extract_query_objects(query_text))
    if extra_objects:
        for eo in extra_objects:
            if eo and isinstance(eo, str):
                query_objects.add(eo.strip().lower())

    ocr_keywords = set(extract_potential_ocr_keywords(query_text))
    if extra_ocr_keywords:
        for ok in extra_ocr_keywords:
            if ok and isinstance(ok, str):
                ocr_keywords.add(ok.strip().lower())

    ocr_indexer = get_ocr_indexer()

    out = []
    for r in raw_results:
        video_id = r.get("video_id", "")
        filename = r.get("filename", "")

        # A. Object Score
        frame_objects_lower = {str(o).lower() for o in r.get("objects", [])}
        matched_obj = 0
        for qo in query_objects:
            qo_clean = qo.lower()
            if qo_clean in frame_objects_lower or any(qo_clean in fo for fo in frame_objects_lower):
                matched_obj += 1

        obj_bonus = min(matched_obj * cfg.object_alpha, cfg.max_object_bonus)

        # B. OCR Score
        ocr_text = r.get("ocr_text") or ocr_indexer.get_text(video_id, filename)
        ocr_bonus = 0.0
        matched_ocr = 0

        if ocr_text and ocr_keywords:
            ocr_text_lower = ocr_text.lower()
            for kw in ocr_keywords:
                if kw in ocr_text_lower:
                    matched_ocr += 1
                    # Khớp chuẩn xác nguyên từ hoặc số
                    if f" {kw} " in f" {ocr_text_lower} " or kw == ocr_text_lower:
                        ocr_bonus += cfg.ocr_alpha
                    else:
                        ocr_bonus += cfg.ocr_alpha * 0.7

        # C. Tổng hợp điểm mới
        base_score = float(r.get("score", 0.0))
        final_score = base_score + obj_bonus + ocr_bonus

        new_r = dict(r)
        new_r["score"] = round(final_score, 4)
        new_r["matched_objects"] = matched_obj
        new_r["matched_ocr"] = matched_ocr
        if ocr_text:
            new_r["ocr_text"] = ocr_text
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


def group_temporal(raw_results: list[dict], cfg: Optional[PipelineConfig] = None) -> list[dict]:
    """Gom cụm theo timestamp; nếu thiếu timestamp (None), dùng frame_id làm
    proxy thứ tự — KHÔNG loại bỏ frame chỉ vì thiếu timestamp."""
    if cfg is None:
        cfg = PipelineConfig()
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
        self,
        event_queries: list[str] | str,
        top_k: int = 100,
        top_k_candidates_per_event: int = 100,
    ) -> list[dict]:
        """
        TRAKE (Temporal Retrieval and Alignment of Key Events):
        Nhận vào danh sách N sự kiện theo thứ tự thời gian.
        Tìm các video chứa chuỗi N sự kiện thỏa mãn: time(f_1) < time(f_2) < ... < time(f_N).
        Trả về danh sách top_k video phù hợp nhất, mỗi video gồm danh sách N frame_ids.
        """
        if isinstance(event_queries, str):
            # Tách chuỗi theo dấu -> hoặc xuống dòng hoặc dấu chấm phẩy
            if "->" in event_queries:
                events = [e.strip() for e in event_queries.split("->") if e.strip()]
            elif "\n" in event_queries:
                events = [e.strip() for e in event_queries.split("\n") if e.strip()]
            elif ";" in event_queries:
                events = [e.strip() for e in event_queries.split(";") if e.strip()]
            else:
                events = [event_queries.strip()]
        else:
            events = [str(e).strip() for e in event_queries if str(e).strip()]

        if not events:
            return []

        N = len(events)

        # 1. Thu thập candidate frames cho từng event
        event_candidates = []
        all_video_ids = set()

        for eq in events:
            raw = self._raw_search(eq, top_k=top_k_candidates_per_event)
            raw = rerank_by_objects(raw, eq, self.config)
            event_candidates.append(raw)
            for r in raw:
                all_video_ids.add(r["video_id"])

        # 2. Xếp chuỗi sự kiện theo thứ tự thời gian cho từng video
        matched_videos = []

        for video_id in all_video_ids:
            # Lấy frames của video này cho từng event
            video_event_frames = []
            for ev_idx in range(N):
                cands = [r for r in event_candidates[ev_idx] if r["video_id"] == video_id]
                for c in cands:
                    c["_ts"] = float(c.get("timestamp") or c.get("frame_id") or 0.0)
                # Sắp xếp theo timestamp tăng dần
                cands.sort(key=lambda x: x["_ts"])
                video_event_frames.append(cands)

            # Quy hoạch động / Greedy tìm chuỗi N frames có t(1) < t(2) < ... < t(N)
            chosen_sequence = []
            last_ts = -1.0
            valid_sequence = True
            total_score = 0.0

            for ev_idx in range(N):
                cands = video_event_frames[ev_idx]
                valid_cands = [c for c in cands if c["_ts"] > last_ts]
                if not valid_cands:
                    valid_sequence = False
                    break
                best_cand = max(valid_cands, key=lambda x: x["score"])
                chosen_sequence.append(best_cand)
                last_ts = best_cand["_ts"]
                total_score += best_cand["score"]

            if valid_sequence and len(chosen_sequence) == N:
                avg_score = round(total_score / N, 4)
                frame_ids = [c["frame_id"] for c in chosen_sequence]
                filenames = [c.get("filename") for c in chosen_sequence]
                matched_videos.append({
                    "video_id": video_id,
                    "frame_ids": frame_ids,
                    "filenames": filenames,
                    "score": avg_score,
                    "events": [
                        {
                            "event": events[i],
                            "frame_id": chosen_sequence[i]["frame_id"],
                            "filename": chosen_sequence[i].get("filename"),
                            "timestamp": chosen_sequence[i]["_ts"],
                            "score": chosen_sequence[i]["score"],
                            "image_path": chosen_sequence[i].get("image_path"),
                        }
                        for i in range(N)
                    ],
                })

        matched_videos.sort(key=lambda x: -x["score"])
        return matched_videos[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# CHUYỂN SANG FORMAT NỘP BÀI
# ═══════════════════════════════════════════════════════════════════════

def to_kis_json(segments: list[dict], max_answers: int = 100) -> list[dict]:
    return [{"video_id": s["video_id"], "frame_id": s["frame_id"]}
            for s in segments if s.get("frame_id") is not None][:max_answers]


def to_qa_json(segments: list[dict], max_answers: int = 100) -> list[dict]:
    return [{"video_id": s["video_id"], "frame_id": s["frame_id"], "answer": s.get("answer", "")}
            for s in segments if s.get("frame_id") is not None][:max_answers]


def to_trake_json(trake_results: list[dict], max_answers: int = 100) -> list[dict]:
    return [{"video_id": t["video_id"], "frame_ids": t["frame_ids"]}
            for t in trake_results if t.get("video_id") and t.get("frame_ids")][:max_answers]


if __name__ == "__main__":
    import time
    from .retriever import Retriever 

    print("⏳ Đang khởi tạo hệ thống và nạp dữ liệu thật...")
    real_retriever = Retriever()
    real_retriever._ensure_loaded()

    pipeline = SearchPipeline(
        retriever=real_retriever,
        config=PipelineConfig(base_threshold=0.20, object_alpha=0.15)
    )

    print("\n" + "="*50)
    print("1️⃣ TEST PHÉP KIS (Known-Item Search)")
    print("="*50)
    query_kis = "người đàn ông đi xe máy"
    t0 = time.time()
    kis_results = pipeline.search_kis(query_kis, top_k=50)
    print(f"⏱️ Thời gian: {time.time() - t0:.2f}s | Tìm thấy: {len(kis_results)} dải sự kiện")
    if kis_results:
        print(f"   Ví dụ top 1: Video {kis_results[0]['video_id']} ({kis_results[0]['start']}s -> {kis_results[0]['end']}s)")

    print("\n" + "="*50)
    print("\n" + "="*50)
    print("2️⃣ TEST PHÉP TRAKE (Temporal Activity)")
    print("="*50)
    # Ví dụ tìm chuỗi hành động tuần tự: bước vào phòng rồi ngồi xuống
    query_trake_steps = ["người bước vào phòng", "người ngồi xuống ghế"]
    t0 = time.time()
    trake_results = pipeline.search_trake(query_trake_steps)
    print(f"⏱️ Thời gian: {time.time() - t0:.2f}s")
    
    # trake_results là list[dict], lấy phần tử đầu tiên
    if trake_results:
        top = trake_results[0]
        print(f"   Ví dụ top 1: Video {top.get('video_id')} theo đúng trật tự thời gian")
        print(f"   Chi tiết các sự kiện: {top.get('events')}")
    else:
        print("   Không tìm thấy chuỗi sự kiện phù hợp.")

    print("\n" + "="*50)
    print("3️⃣ TEST PHÉP Q&A (Question Answering)")
    print("="*50)
    query_qa = "ô tô màu đỏ"
    question_qa = "có bao nhiêu chiếc ô tô màu đỏ trong video"
    
    t0 = time.time()
    # Sửa tại đây: Truyền đủ 2 tham số query_text và question
    qa_result = pipeline.search_qa(query_text=query_qa, question=question_qa)
    
    print(f"⏱️ Thời gian: {time.time() - t0:.2f}s")
    print(f"   Kết quả trả về JSON chuẩn bị nộp: {qa_result}")
    print("="*50)