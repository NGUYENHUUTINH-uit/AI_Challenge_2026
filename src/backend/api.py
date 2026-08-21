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
from src.retrieval.search_pipeline import SearchPipeline, PipelineConfig, rerank_by_objects, group_temporal

app = FastAPI(
    title="AI Challenge 2026 - Video Retrieval API",
    description="Backend API cho tim kiem video KIS, Q&A va TRAKE",
    version="2.0.0"
)

# Cho phep UI (file:// hoac localhost) goi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Retriever \u2014 load FAISS + CLIP 1 lan khi khoi dong
_retriever = Retriever()


# ============================================================
# SCHEMAS
# ============================================================

class TextQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 20
    session_id: Optional[str] = None    # danh cho Conversational KIS


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "AIC 2026 Retrieval API dang hoat dong",
        "version": "2.0.0"
    }


@app.post("/api/v1/search/text")
async def search_by_text(request: TextQueryRequest):
    """
    Tim kiem keyframe bang text query.
    1. CLIP encode query -> vector
    2. FAISS search -> top-K frames
    3. Object re-rank -> sap xep lai theo object match
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query khong duoc de trong.")

        # 1. FAISS search
        raw_results = _retriever.search_keyframes(
            query_text=request.query,
            top_k=request.top_k * 2    # lay du de re-rank
        )

        # 2. Object re-rank (boost score neu frame co object khop query)
        reranked = rerank_by_objects(raw_results, request.query)

        # 3. Tra ve top_k ket qua cuoi
        final = reranked[: request.top_k]

        return {
            "status":  "success",
            "query":   request.query,
            "total":   len(final),
            "results": final,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loi tim kiem: {str(e)}")


@app.post("/api/v1/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 20):
    """Tim kiem bang anh upload. (TODO: implement CLIP image encode)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chi chap nhan file hinh anh.")
    return {
        "status": "todo",
        "message": "Image search chua implement. Dang su dung text search.",
        "results": []
    }


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