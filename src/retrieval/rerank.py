def extract_objects_from_query(query_text):
    """
    Giả lập việc trích xuất danh sách các objects/thực thể từ câu query thô.
    Trong thực tế, có thể dùng NLP, NER hoặc LLM.
    """
    query_lower = query_text.lower()
    objects = []
    
    # Từ điển giả lập
    keywords_map = {
        "người": "person",
        "đàn ông": "person",
        "phụ nữ": "person",
        "xe đạp": "bicycle",
        "xe hơi": "car",
        "ô tô": "car",
        "chó": "dog",
        "mèo": "cat",
        "cây": "tree",
    }
    
    for vi_word, en_word in keywords_map.items():
        if vi_word in query_lower:
            objects.append(en_word)
            
    return list(set(objects))

def rerank_by_objects(faiss_results, query_text):
    """
    Nhận kết quả thô từ FAISS và lọc/tái sắp xếp thứ tự ưu tiên 
    dựa trên sự xuất hiện của các thực thể/vật thể.
    
    Args:
        faiss_results (list): Danh sách dictionary kết quả từ FAISS.
            Mỗi dict cần có trường 'objects' (list các objects có trong frame) và 'score'.
        query_text (str): Câu truy vấn gốc.
        
    Returns:
        list: Danh sách kết quả đã được tái sắp xếp.
    """
    query_objects = extract_objects_from_query(query_text)
    print(f"[Reranker] Các objects tìm được từ query: {query_objects}")
    
    if not query_objects:
        # Nếu không trích xuất được object nào, giữ nguyên kết quả
        return faiss_results
        
    reranked_results = []
    
    for res in faiss_results:
        frame_objects = res.get("objects", [])
        
        # Đếm số lượng objects khớp giữa query và frame
        match_count = sum(1 for obj in query_objects if obj in frame_objects)
        
        # Tính toán điểm số mới (Boost score dựa trên objects matching)
        # Ví dụ: mỗi object khớp sẽ cộng thêm 0.1 vào score ban đầu
        original_score = res.get("score", 0.0)
        new_score = original_score + (match_count * 0.1)
        
        # Cập nhật thông tin
        new_res = res.copy()
        new_res["original_score"] = original_score
        new_res["score"] = new_score
        new_res["matched_objects"] = match_count
        
        reranked_results.append(new_res)
        
    # Sắp xếp lại dựa trên điểm số mới
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    return reranked_results

if __name__ == "__main__":
    # Test script
    dummy_results = [
        {"video_id": "v1", "frame_idx": 100, "score": 0.85, "objects": ["person", "car"]},
        {"video_id": "v2", "frame_idx": 200, "score": 0.90, "objects": ["tree", "building"]},
        {"video_id": "v3", "frame_idx": 300, "score": 0.80, "objects": ["person", "bicycle", "dog"]}
    ]
    
    query = "Một người đàn ông và chiếc xe đạp"
    
    print("--- Trước khi Rerank ---")
    for r in dummy_results:
        print(f"{r['video_id']}: score={r['score']:.2f}, objects={r['objects']}")
        
    reranked = rerank_by_objects(dummy_results, query)
    
    print("\n--- Sau khi Rerank ---")
    for r in reranked:
         print(f"{r['video_id']}: new_score={r['score']:.2f} (original={r['original_score']:.2f}), matched={r['matched_objects']}, objects={r['objects']}")
