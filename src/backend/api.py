import os
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.backend.fallback import run_fallback_search

app = FastAPI(
    title="AI Challenge API 2026",
    description="Backend pipeline hỗ trợ phân tích và truy xuất dữ liệu lớn multimedia",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


@app.get("/")
async def health_check():
    return {"status": "ok"}


class SearchRequest(BaseModel):
    text: str
    history: Optional[List[dict]] = Field(default_factory=list)
    session_id: Optional[str] = None


# C2: backend không nên trả mock data nữa. Thay vào đó, backend nên gọi
# pipeline của người B (RetrievalAgent + temporal localization) để nhận kết quả.
try:
    from src.retrieval.adapter import MetadataLookup, make_search_fn
    from src.retrieval.agent import AgentConfig, RetrievalAgent
    from src.retrieval.retriever import Retriever
    from src.retrieval.temporal import TemporalConfig
except Exception as exc:  # pragma: no cover - fallback path khi module B chưa sẵn sàng
    MetadataLookup = None
    make_search_fn = None
    RetrievalAgent = None
    AgentConfig = None
    Retriever = None
    TemporalConfig = None
    RETRIEVAL_IMPORT_ERROR = str(exc)
else:
    RETRIEVAL_IMPORT_ERROR = None


SESSION_AGENTS: Dict[str, object] = {}


def _build_agent() -> object:
    """Khởi tạo agent từ module B bằng cách dùng adapter của người D."""
    if RetrievalAgent is None or Retriever is None or make_search_fn is None:
        raise RuntimeError(f"Retrieval modules unavailable: {RETRIEVAL_IMPORT_ERROR}")

    metadata_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "metadata.json")
    )
    metadata = MetadataLookup(metadata_path)
    retriever = Retriever(use_dummy=True)
    search_fn = make_search_fn(retriever, metadata=metadata, apply_rerank=True)
    config = AgentConfig(temporal_config=TemporalConfig())
    return RetrievalAgent(search_fn=search_fn, config=config)


def _get_or_create_agent(session_id: str) -> object:
    # Mỗi session_id nên có 1 agent riêng để giữ memory/context cho KISC.
    if session_id not in SESSION_AGENTS:
        SESSION_AGENTS[session_id] = _build_agent()
    return SESSION_AGENTS[session_id]


@app.post("/search")
async def search(query: SearchRequest):
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query text cannot be empty")

    session_id = query.session_id or "default"

    try:
        agent = _get_or_create_agent(session_id)
        result = agent.run(query.text.strip())
    except Exception as exc:
        fallback_result = run_fallback_search(query.text.strip(), top_k=5)
        return {
            "status": "fallback",
            "query_processed": query.text,
            "session_id": session_id,
            "segments": fallback_result["segments"],
            "results": fallback_result["segments"],
            "used_fallback": True,
            "error": str(exc),
        }

    segments = []
    for segment in result.get("segments", []):
        segments.append(
            {
                "video_id": segment.get("video_id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "avg_similarity": segment.get("avg_similarity"),
                "peak_similarity": segment.get("peak_similarity"),
                "n_frames": segment.get("n_frames"),
            }
        )

    return {
        "status": "success",
        "query_processed": query.text,
        "session_id": session_id,
        "action": result.get("action"),
        "resolved_query": result.get("resolved_query"),
        "segments": segments,
        "results": segments,
        "latency_seconds": result.get("latency_seconds"),
    }


@app.post("/api/v1/search/text")
async def search_by_text(request: SearchRequest):
    # Endpoint này giữ lại như alias tương thích ngược cho UI/API cũ.
    return await search(request)


@app.post("/search-image")
async def search_image(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only PNG and JPG are allowed.")

    # Lưu file tạm thời để xử lý; trong tương lai có thể thay bằng encoder ảnh thực tế.
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    os.remove(temp_file_path)
    return {
        "status": "accepted",
        "file_processed": file.filename,
        "segments": [],
        "note": "Image search is currently a placeholder; connect to image encoder and retrieval pipeline later.",
    }