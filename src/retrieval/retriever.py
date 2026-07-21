import random

class Retriever:
    def __init__(self, use_dummy=True):
        """
        Khởi tạo Retriever. 
        Nếu use_dummy=True, sử dụng dữ liệu giả lập thay vì load model thật.
        """
        self.use_dummy = use_dummy
        if not self.use_dummy:
            # TODO: Load FAISS index và CLIP model thật ở đây
            pass
            
    def _dummy_embedding(self, query):
        """Hàm giả lập việc tạo vector embedding cho query."""
        # Trả về một vector giả lập có chiều dài 512 (giống CLIP)
        return [random.random() for _ in range(512)]
        
    def _dummy_search(self, embedding, top_k):
        """Hàm giả lập việc search vector trên FAISS."""
        results = []
        for i in range(top_k):
            # Tạo dữ liệu giả lập cho kết quả trả về
            video_id = f"video_{random.randint(1, 100):03d}"
            frame_idx = random.randint(0, 5000)
            score = random.uniform(0.5, 0.99)
            results.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "score": score,
                "image_path": f"data/frames/{video_id}/{frame_idx:06d}.jpg",
                "objects": ["person", "car", "dog"] if random.random() > 0.5 else ["tree", "building"]
            })
        # Sắp xếp theo score giảm dần
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_keyframes(self, query_text=None, query_image=None, top_k=100):
        """
        Hàm search vector cơ bản nhận đầu vào đa phương thức (Text và Image).
        
        Args:
            query_text (str): Câu truy vấn bằng văn bản.
            query_image (str hoặc PIL.Image): Đường dẫn tới ảnh hoặc object ảnh.
            top_k (int): Số lượng kết quả muốn trả về.
            
        Returns:
            list: Danh sách các keyframe tìm được kèm theo điểm số.
        """
        print(f"Start searching with Text: '{query_text}', Image: '{query_image}'")
        
        # 1. Chuyển đổi query thành vector embedding
        if self.use_dummy:
            embedding = self._dummy_embedding(query_text or query_image)
        else:
            # TODO: Xử lý embedding thật bằng CLIP
            pass
            
        # 2. Tìm kiếm trên Vector Database (VD: FAISS)
        if self.use_dummy:
            results = self._dummy_search(embedding, top_k)
        else:
            # TODO: Tìm kiếm thật trên FAISS
            pass
            
        print(f"Found {len(results)} results.")
        return results

if __name__ == "__main__":
    # Test thử hàm
    retriever = Retriever(use_dummy=True)
    res = retriever.search_keyframes(query_text="A man riding a bicycle", top_k=5)
    for r in res:
        print(r)
