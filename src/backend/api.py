from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="AI Challenge 2026 - Retrieval API",
    description="Backend API phục vụ truy xuất Video/Image KIS và Conversational KIS",
    version="1.0.0"
)

class TextQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    session_id: Optional[str] = None

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Hệ thống AI Challenge 2026 API đang hoạt động mượt mà!"}

@app.post("/api/v1/search/text")
async def search_by_text(request: TextQueryRequest):
    try:
        return {
            "status": "success",
            "query_received": request.query,
            "results": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận: {str(e)}")

@app.post("/api/v1/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 10):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Vui lòng chỉ tải lên file định dạng hình ảnh.")
    try:
        return {
            "status": "success",
            "filename": file.filename,
            "results": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")