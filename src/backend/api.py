"""
api.py \u2014 FastAPI Backend cho AIC 2026 Video Retrieval
=======================================================
Endpoints:
  GET  /                       health check
  POST /api/v1/search/text     tim kiem bang text query
  POST /api/v1/search/image    tim kiem bang anh upload  (TODO)
  GET  /api/v1/image/{path}    phuc vu anh keyframe cho UI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from src.retrieval.retriever import Retriever
from src.retrieval.search_pipeline import (
    SearchPipeline, PipelineConfig, rerank_by_objects, group_temporal,
    to_kis_json, to_qa_json, to_trake_json
)

app = FastAPI(
    title="AI Challenge 2026 - Video Retrieval API",
    description="Backend API cho 3 bai toan: Textual KIS, Visual Q&A va TRAKE",
    version="2.1.0"
)

# Cho phep UI (file:// hoac localhost) goi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Retriever & SearchPipeline
_retriever = Retriever()
_pipeline = SearchPipeline(retriever=_retriever)


# ============================================================
# SCHEMAS
# ============================================================

class KisQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 100
    session_id: Optional[str] = None


class QaQueryRequest(BaseModel):
    query: str
    question: Optional[str] = None
    top_k: Optional[int] = 100


class TrakeQueryRequest(BaseModel):
    events: list[str] | str
    top_k: Optional[int] = 100


class AnswerGenerateRequest(BaseModel):
    video_id: str
    filename: str
    question: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "AIC 2026 Retrieval API dang hoat dong",
        "supported_modes": ["kis", "qa", "trake"],
        "version": "2.1.0"
    }


# --- 1. DẠNG BÀI 1: TEXTUAL KIS ---
@app.post("/api/v1/search/kis")
@app.post("/api/v1/search/text")
async def search_kis(request: KisQueryRequest):
    """
    Truy vấn Textual KIS (Known Item Search).
    Format nộp bài CSV: <video_id>,<frame_id> (tối đa 100 dòng).
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query không được để trống.")

        raw_results = _retriever.search_keyframes(
            query_text=query,
            top_k=min(request.top_k * 2, 200)
        )

        reranked = rerank_by_objects(raw_results, query)
        final = reranked[: request.top_k]

        return {
            "status": "success",
            "mode": "kis",
            "query": query,
            "total": len(final),
            "results": final,
            "submission_csv": "\n".join(f"{r['video_id']},{r['frame_id']}" for r in final[:100])
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm KIS: {str(e)}")


# --- 2. DẠNG BÀI 2: VISUAL QUESTION ANSWERING (Q&A) ---
@app.post("/api/v1/search/qa")
async def search_qa(request: QaQueryRequest):
    """
    Truy vấn Visual Question Answering (Q&A).
    Format nộp bài CSV: <video_id>,<frame_id>,"<answer>" (tối đa 100 dòng).
    """
    try:
        query = request.query.strip()
        question = (request.question or query).strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query hoặc câu hỏi không được để trống.")

        raw_results = _retriever.search_keyframes(
            query_text=query,
            top_k=min(request.top_k * 2, 200)
        )

        reranked = rerank_by_objects(raw_results, query)
        final = reranked[: request.top_k]

        # Sinh gợi ý đáp án sơ bộ cho từng frame (dựa trên objects / OCR)
        for r in final:
            predicted_ans = ""
            objs = r.get("objects", [])
            ocr = r.get("ocr_text", "")
            q_lower = question.lower()

            if "bao nhiêu" in q_lower or "mấy" in q_lower or "count" in q_lower:
                for target in ["người", "person", "xe", "car", "motorcycle", "chó", "dog"]:
                    if target in q_lower:
                        cnt = sum(1 for o in objs if target in o.lower())
                        if cnt > 0:
                            predicted_ans = str(cnt)
                            break
                if not predicted_ans and objs:
                    predicted_ans = str(len(objs))
            elif "chữ" in q_lower or "biển" in q_lower or "số" in q_lower or "text" in q_lower:
                if ocr:
                    predicted_ans = ocr[:50]
            elif "màu" in q_lower or "color" in q_lower:
                for color in ["đỏ", "xanh", "vàng", "trắng", "đen", "cam", "hồng", "tím"]:
                    if color in q_lower:
                        predicted_ans = f"Màu {color}"
                        break

            r["answer"] = predicted_ans

        # Tạo CSV preview
        csv_lines = []
        for r in final[:100]:
            ans = r.get("answer", "")
            if ans and ("," in ans or '"' in ans or "\n" in ans):
                ans_clean = f'"{ans.replace(chr(34), chr(34)+chr(34))}"'
            elif ans:
                ans_clean = f'"{ans}"'
            else:
                ans_clean = '""'
            csv_lines.append(f"{r['video_id']},{r['frame_id']},{ans_clean}")

        return {
            "status": "success",
            "mode": "qa",
            "query": query,
            "question": question,
            "total": len(final),
            "results": final,
            "submission_csv": "\n".join(csv_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm Q&A: {str(e)}")


# --- 3. DẠNG BÀI 3: TRAKE (TEMPORAL ALIGNMENT) ---
@app.post("/api/v1/search/trake")
async def search_trake_endpoint(request: TrakeQueryRequest):
    """
    Truy vấn TRAKE (Temporal Retrieval and Alignment of Key Events).
    Format nộp bài CSV: <video_id>,<frame_1>,<frame_2>,...,<frame_N> (tối đa 100 dòng).
    """
    try:
        matched_videos = _pipeline.search_trake(
            event_queries=request.events,
            top_k=request.top_k
        )

        csv_lines = []
        for v in matched_videos[:100]:
            frames_str = ",".join(str(fid) for fid in v["frame_ids"])
            csv_lines.append(f"{v['video_id']},{frames_str}")

        return {
            "status": "success",
            "mode": "trake",
            "events": request.events if isinstance(request.events, list) else [request.events],
            "total": len(matched_videos),
            "results": matched_videos,
            "submission_csv": "\n".join(csv_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm TRAKE: {str(e)}")


@app.get("/api/v1/image/{video_id}/{filename}")
async def serve_keyframe(video_id: str, filename: str):
    """
    Phuc vu file anh keyframe cho UI hien thi.
    Tu dong tim kiem cac bien the ten file (001.jpg, 0001.jpg, 0000.jpg...)
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR     = PROJECT_ROOT / "data" / "processed"
    BTC_DIR      = PROJECT_ROOT / "data" / "btc"

    # Danh sach cac thu muc chua keyframe
    search_dirs = [
        BTC_DIR  / "Keyframes"       / video_id,
        DATA_DIR / "keyframes"       / video_id,
        DATA_DIR / "keyframes_clean" / video_id,
    ]

    # Tao danh sach cac bien the ten file co the co
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    name_variants = [filename]

    try:
        idx = int(stem)
        name_variants.extend([
            f"{idx:03d}{suffix}",      # BTC format: 001.jpg
            f"{idx:04d}{suffix}",      # 4-digit: 0001.jpg
            f"{idx + 1:03d}{suffix}",  # neu request 0-indexed (0000 -> 001.jpg)
            f"{idx + 1:04d}{suffix}",
            f"{idx - 1:03d}{suffix}",  # neu request 1-indexed (001 -> 000.jpg)
            f"{idx - 1:04d}{suffix}",
            f"{idx}{suffix}",
        ])
    except ValueError:
        pass

    # Tim kiem file theo thu tu
    for vdir in search_dirs:
        if not vdir.exists():
            continue
        for name in name_variants:
            img_path = vdir / name
            if img_path.exists() and img_path.is_file():
                return FileResponse(str(img_path), media_type="image/jpeg")

    raise HTTPException(
        status_code=404,
        detail=f"Khong tim thay anh: {video_id}/{filename}"
    )