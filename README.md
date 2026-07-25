# Kế hoạch v4 — Đội 4 người AI Challenge 2026 (Cập nhật 12/7/2026)

**Cấu trúc hệ thống:** (image data → AI models): indexing → databases → (UI → User): retrieval

## 1. Phạm vi & Mục tiêu Chiến lược

- **Bài toán cốt lõi**: Xây dựng hệ thống Trợ lý ảo thông minh giải quyết trọn vẹn các dạng bài toán:
  - **Textual KIS**: Truy xuất video từ câu mô tả văn bản thô.
  - **Video/Image KIS**: Truy xuất video gốc từ một đoạn video ngắn hoặc hình ảnh/sketch được cung cấp (Image-to-Video Retrieval).
  - **Conversational KIS (KISC)**: Tìm kiếm thông qua hội thoại nhiều lượt, tận dụng ngữ cảnh lịch sử trò chuyện để thu hẹp phạm vi tìm kiếm.

- **Đặc điểm dữ liệu**: Xử lý dữ liệu đa phương tiện multimedia có khối lượng cực lớn (Big Data), không cấu trúc (Unstructured) từ nguồn _Surveillance_ (camera an ninh) và _Sousveillance_ (camera hành trình, kính thông minh). Dữ liệu có độ nhiễu cao (góc quay rung lắc, ánh sáng thay đổi liên tục, nhiễu môi trường) nhưng cực kỳ giàu ngữ nghĩa cá nhân và hành động chuỗi.

- **Thời gian thực hiện**: ~4 tuần (12/07/2026 – 10/08/2026), bám sát lộ trình kỹ thuật và bấm giờ thi thử liên tục.

## 2. Cấu trúc Đội hình & Nguyên tắc Phối hợp

- **Kiến trúc phân lớp**: Hệ thống tuân thủ mô hình:
  $$\text{Image/Video Data} \rightarrow \text{AI Preprocessing \& Models (Stage 1: Indexing)} \rightarrow \text{Vector DB} \longleftrightarrow \text{FastAPI Backend} \longleftrightarrow \text{Advanced UI (Stage 2: Retrieval)} \rightarrow \text{User}$$

- **Phân vai trò**:
  - **A — Data & Indexing Engineer**: Lớp 1-2 (Xử lý thô, Tiền xử lý dữ liệu và Lập chỉ mục).
  - **B — AI & Advanced Retrieval Engineer**: Lớp 3 nâng cao (Trọng tâm: Logic điều phối của AI Agent, Temporal Localization, Khai phá/Khám phá ngữ nghĩa).
  - **C — Backend & Integration Engineer**: Lớp 4 (Xây dựng API, Ghép nối hệ thống, Đóng gói, Deploy và xử lý lỗi/fallback).
  - **D — Retrieval Support + Frontend + Test**: Lớp 5 (Phát triển UI tương tác thông minh, phụ trách bộ câu hỏi luyện tập và hỗ trợ B xây dựng các hàm tìm kiếm cơ bản).

- **Nguyên tắc Git**: Tránh xung đột code bằng cách chia nhánh riêng biệt. D và B cùng làm việc trên tầng Retrieval; D commit các hàm tìm kiếm nền tảng vào file `retriever.py`, B import và gọi các hàm này từ file `agent.py`. Không chỉnh sửa chung một file cùng một lúc.

## 3. Bảng Công cụ Mô hình & Phân công Chi tiết

| Công cụ / Mô hình | Vai trò & Nhiệm vụ cụ thể | Phụ trách chính | Tính chất |
| :--- | :--- | :--- | :--- |
| **CLIP (ViT-B/32+)** | Encode cả Keyframe ảnh lẫn Text/Image Query vào chung không gian vector để phục vụ cả Textual KIS và Video KIS. | A (Encode ảnh), B/D (Query) | **Bắt buộc** |
| **Multilingual CLIP / Translation** | Xử lý các câu hỏi truy vấn bằng tiếng Việt (So sánh phương án dịch máy vs Mô hình đa ngôn ngữ). | B, D | **Bắt buộc** |
| **FAISS / Milvus** | Cấu trúc cơ sở dữ liệu vector. Lưu trữ đặc trưng hình ảnh tích hợp siêu dữ liệu (Metadata) phục vụ tìm kiếm tương đồng nhanh (IVF Index). | A (Build), B/D (Query) | **Bắt buộc** |
| **Object Detection (FasterRCNN+)** | Trích xuất danh mục vật thể (600 categories) làm căn cứ chạy hàm re-rank hoặc gợi ý concept phân biệt. | D (Chính), B | **Bắt buộc** |
| **LLM (Claude/GPT/Gemini API)** | Làm nhiệm vụ mở rộng truy vấn (Query Expansion) và làm bộ não suy luận (Reasoning, Planning, Memory) cho AI Agent. | B (Agent), D (Prompt) | **Bắt buộc** |
| **PaddleOCR** | Đọc ký tự văn bản xuất hiện trong keyframe (biển hiệu, chữ viết, phụ đề) để tăng cường thuộc tính tìm kiếm. | C | Tùy chọn (Tuần 3) |
| **Whisper** | Nhận dạng lời thoại, tiếng động môi trường trong video phòng trường hợp đề bài chứa các manh mối âm thanh. | C | Tùy chọn (Tuần 3) |
| **FastAPI** | Đóng gói toàn bộ pipeline thành các cổng API ổn định, chịu tải, hỗ trợ nhận cả text query lẫn file upload (image query). | C | **Bắt buộc** |
| **React / HTML + JS thuần** | Giao diện điều khiển tương tác (Multi-turn chat cho KISC, ô tải ảnh cho Video KIS, hệ thống thẻ gợi ý Concept). | D | **Bắt buộc** |

**Bổ sung**: Preprocessing pipeline (filtering → normalization → grouping scenes) cho A+D.

## 4. Lộ trình chi tiết theo giai đoạn

### GIAI ĐOẠN 1: Xây Dựng Xương Sống Hệ Thống (12/07 – 24/07)

| Task | Mô tả chi tiết | Input | Output | Thời gian | Chính | Hỗ trợ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | Setup môi trường + gom video mẫu (MSR-VTT + tự thu) | Video có sẵn / MSR-VTT | data/videos/ | 12-13/7 | A | - |
| **A2** | `ingest.py` (trích keyframe + metadata) | Video mẫu | Keyframes + metadata.json | 14-15/7 | A | D (từ 25/7) |
| **A3** | `indexer.py` (CLIP embedding + FAISS) | Keyframes | clip_features.npy, index.faiss | 16-18/7 | A | - |
| **A4** | Đo tốc độ, tối ưu batch processing | Pipeline A2+A3 | Bảng đo thời gian | 19-21/7 | A | - |
| **A5** | Preprocessing (filter noisy, grouping scenes) + load switch | Pipeline A | Dataset sạch | 22-24/7 | A | D (25-27/7) |
| **B4** | `find_time_range()` nâng cao (threshold + temporal grouping) | Keyframes + timestamps | Hàm temporal localization | 12-16/7 | B | D |
| **B5** | `agent.py` (Reasoning, Planning, Memory, ReAct, Tool-use) | Query thô | Pipeline Agent hoàn chỉnh | 17-24/7 | B | D (test) |
| **D-B1** | `retriever.py` cơ bản | clip_features.npy | search_keyframes() | 12-13/7 | D | B |
| **D1** | Xây `query_test_set.md` (≥30 câu, bao conversational) | Slide + ví dụ | File test queries | 12-13/7 | D | - |
| **D-B2** | `query_expansion.py` (prompt engineering) | Query gốc | 3-5 variants + keywords | 14-16/7 | D | B |
| **D-B3** | `rerank_by_objects()` | Kết quả thô + Objects | Hàm re-rank | 17-18/7 | D | B |
| **D2** | UI tối giản (input → results) | JSON schema | index.html | 19-21/7 | D | C |
| **C1-C3** | FastAPI framework, nối retriever, test tự động | - | api.py + test | 12-18/7 | C | - |
| **C4** | Deploy đúng máy thi, tối ưu môi trường | Hệ thống chạy trên máy dev | `setup.sh` hoàn chỉnh | 19-21/7 | C | - |
| **C5** | Xử lý lỗi ngoại lệ & cấu hình Timeout | `api.py` hiện tại | Fallback logic hoàn chỉnh | 22-24/7 | C | - |

#### Người A — Data & Indexing Stage
- **A1 (12-13/07)**: Cài đặt môi trường, cấu trúc thư mục chuẩn `aic2026/src/...`. Thu thập tập video mẫu (MSR-VTT) kết hợp video tự quay định dạng nhiễu để giả lập môi trường surveillance/sousveillance.
- **A2 (14-15/07)**: Viết file `ingest.py` thực hiện trích xuất keyframe thô và ghi nhận siêu dữ liệu thời gian (metadata.json).
- **A3 (16-18/07)**: Viết file `indexer.py` ứng dụng CLIP trích xuất vector đặc trưng `clip_features.npy` và tạo lập cấu trúc lưu trữ cơ bản bằng FAISS (`index.faiss`).
- **A4 (19-21/07)**: Đo đạc tốc độ lập chỉ mục, tinh chỉnh cơ chế xử lý theo mẻ (Batch Processing) để giải quyết bài toán cân bằng giữa: _Tốc độ chạy - Sức mạnh thuật toán - Chi phí hệ thống_.
- **A5 (22-24/07)**: Thiết lập Pipeline Tiền xử lý nâng cao: Lọc frame nhòe/nhiễu ánh sáng do chuyển động (**Filtering**) $\rightarrow$ Chuẩn hóa định dạng vector (**Normalization**) $\rightarrow$ Gom cụm các frame tương đồng trong cùng một video thành các phân cảnh/sự kiện đặc trưng (**Grouping Scenes**). Cài đặt công tắc chuyển đổi tập dữ liệu nhanh (`load_dataset(source=...)`).

#### Người B — Advanced Retrieval Layer
- **B4 (12-16/07)**: Phát triển hàm `find_time_range()` nâng cao. Không sử dụng khoảng thời gian cố định mà tính toán tương đồng theo ngưỡng biến thiên (Similarity Thresholding) kết hợp gom cụm thời gian để định vị chính xác phân đoạn video giàu ngữ nghĩa cá nhân.
- **B5 (17-24/07)**: Phát triển bộ não điều phối `agent.py` dựa trên các nền tảng của AI Agent: **Reasoning** (Suy luận ngữ cảnh chuỗi), **Planning** (Lập kế hoạch gọi tool thích hợp: rập khuôn retrieval, mở rộng từ khóa, re-rank), và **Memory** (Lưu vết lịch sử chat để xử lý bài toán KISC nhiều lượt). Triển khai cấu trúc ReAct hoặc chuỗi suy luận CoT/ToT rút gọn.

#### Người D — Retrieval Support & Smart UI
- **D-B1 (12-13/07)**: Khởi tạo file `retriever.py` viết hàm search vector cơ bản `search_keyframes()` nhận đầu vào đa phương thức (chấp nhận cả Text Query và Image Query chuyển đổi qua CLIP).
- **D1 (12-13/07)**: Thiết lập tập câu hỏi thi thử `query_test_set.md` ($\ge 30$ câu). Thiết kế đa dạng cho cả bài toán KIS mô tả thô lẫn các đoạn hội thoại đa lượt phức tạp của bài toán KISC.
- **D-B2 (14-16/07)**: Viết module `query_expansion.py` sử dụng Prompt Engineering trên LLM để tự động sinh 3-5 biến thể truy vấn chi tiết từ câu hỏi thô của người dùng.
- **D-B3 (17-18/07)**: Phát triển hàm `rerank_by_objects()`. Nhận kết quả thô từ FAISS và tiến hành lọc/tái sắp xếp thứ tự ưu tiên dựa trên sự xuất hiện của các thực thể/vật thể (Objects dữ liệu giả lập).
- **D2 (19-21/07)**: Thiết kế wireframe và code giao diện `index.html` tối giản. UI phải phân tách rõ hai cơ chế xử lý phản hồi:
  - **Cơ chế Khám phá (Exploration)**: Hiển thị các thông tin diện rộng, gợi ý các Concept phụ mở rộng phạm vi tìm kiếm khi người dùng diễn đạt chưa chặt chẽ.
  - **Cơ chế Khai phá (Exploitation)**: Hiển thị thông tin mức độ liên quan cực cao, gợi ý các Concept phân biệt (ví dụ đặc điểm màu sắc, hành động) nhằm tách bạch các phân cảnh có độ giống nhau lớn, giảm sự không chắc chắn của mô hình.

#### Người C — Backend & Integration Layer
- **C1 (12-13/07)**: Thiết lập bộ khung API FastAPI rỗng (`api.py`), cấu hình các endpoint cơ bản chạy ổn định.
- **C2 (14-16/07)**: Kết nối API `/search` với file nền tảng `retriever.py` của D để nhận diện query và trả kết quả dưới dạng JSON chuẩn. Endpoint hỗ trợ nhận cả chuỗi ký tự lẫn file ảnh tải lên.
- **C3 (17-18/07)**: Viết kịch bản kiểm thử tự động hệ thống API (`test_api.py`) bằng thư viện Pytest để bảo đảm không xảy ra lỗi query rỗng hoặc timeout ngoài ý muốn.
- **C4 (19-21/07)**: Viết script đóng gói tự động (`setup.sh`). Thử nghiệm deploy toàn bộ hệ thống lên máy cấu hình tương đương máy thi đấu để kiểm tra tính tương thích của thư viện, phần cứng và docker (nếu có).
- **C5 (22-24/07)**: Xây dựng cơ chế bắt lỗi (`try/except`) và xử lý nghẽn/timeout. Cài đặt logic Fallback khẩn cấp: Nếu module AI Agent suy luận (B5) bị quá thời gian hoặc crash, hệ thống tự động bypass qua Agent và chuyển sang dùng hàm tìm kiếm vector thô (`search_keyframes`) để luôn đảm bảo trả về kết quả JSON cho BTC đúng hạn.

> **Điểm đồng bộ quan trọng Cuối Giai đoạn A (24/07)**: Thực hiện chạy thử nghiệm liên chuỗi khép kín $A \rightarrow B \rightarrow C \rightarrow D$ trên bộ dữ liệu mẫu. Dùng API thật để kết nối UI, nhập 5-10 câu test thật để kiểm tra khả năng tích hợp.

### GIAI ĐOẠN 2: Tích Hợp Dữ Liệu Thật & Luyện Tập Thi Đấu (25/07 – 10/08)

| Ngày | Việc chính | Phụ trách | Ghi chú |
| :--- | :--- | :--- | :--- |
| 25-27/7 | Indexing data thật (khối lượng lớn) | A + D | D hỗ trợ A mạnh |
| 28-30/7 | Test retriever + Agent trên data thật | B + D | - |
| 29/7-2/8 | Tinh chỉnh Agent, threshold, Memory/ReAct | B (chính) + D (test) | - |
| 3-6/8 | Full integration, polish UI (gợi ý concepts, cluster, exploration/exploitation) | Cả đội (D UI chính) | - |
| 7-8/8 | Thi thử bấm giờ (≥3 lần) | Cả đội | Bao conversational |
| 9/8 | Đóng gói, backup, fallback | C | - |
| 10/8 | Nộp bài | Cả đội | - |

- **25-27/07: Lập chỉ mục thần tốc trên dữ liệu thật**
  - **A + D** cùng phối hợp. Khi Ban tổ chức bàn giao kho dữ liệu lớn, A vận hành hệ thống pipeline đã tối ưu hóa, D hỗ trợ kiểm tra định dạng lỗi, chạy song song các luồng xử lý video để bóc tách keyframe, lọc nhiễu và đóng gói index FAISS hoàn chỉnh.
- **28-30/07: Kiểm thử tổng lực trên Data thật**
  - **B + D** thực hiện chạy thử nghiệm bộ cấu trúc Agent và Retriever trên chỉ mục thật vừa tạo, đánh giá độ sai lệch ngữ nghĩa và ghi nhận các lỗ hổng tìm kiếm.
- **29/07 – 02/08: Tinh chỉnh tối ưu hóa Thuật toán**
  - **B** đóng vai trò chủ lực tinh chỉnh các tham số ngưỡng Threshold của hàm định vị thời gian, tối ưu hóa bộ nhớ ngắn hạn/dài hạn (Memory) của Agent khi phản hồi các câu hỏi dạng KISC. **D** thực hiện chạy các kịch bản kiểm thử song song.
- **03-06/08: Hoàn thiện Giao diện Tương tác thông minh (Full Integration)**
  - **Cả đội** tập trung tối ưu hóa UI/UX.
  - **D** hoàn thiện tính năng hiển thị kết quả theo cụm phân cảnh thông minh (tránh tràn ngập các frame trùng lặp của cùng một video), tích hợp lưới hiển thị thẻ gợi ý Concept Khai phá/Khám phá, tối ưu trình phát video nhảy chính xác đến timestamp kết quả.
  - **C** hoàn thiện cơ chế bắt lỗi hệ thống và fallback logic (nếu Agent gặp lỗi suy luận, tự động chuyển về search thô dựa trên tương đồng vector).
- **07-08/08: Thi thử bấm giờ áp lực thực tế**
  - **Cả đội** tham gia thi thử nghiêm túc ít nhất 3 lần sử dụng bộ query nâng cao do **D** chuẩn bị. Phân định rõ vai trò tác chiến lúc thi thật: Người đảm nhiệm đọc đề/phân tích, người gõ query/tương tác chat với Agent, người chịu trách nhiệm kiểm tra hình ảnh trực quan trên UI.
- **09/08: Đóng gói sản phẩm & Dự phòng rủi ro**
  - **C** phụ trách đóng gói toàn bộ hệ thống bằng tập lệnh triển khai `setup.sh`, thiết lập các phương án backup khởi động lại hệ thống cực nhanh trong vòng 30 giây nếu xảy ra sự cố sập nguồn hoặc crash ứng dụng khi đang thi đấu.
- **10/08**: Hoàn tất thủ tục nộp bài.

## 5. Checklist Kiểm Tra Điều Kiện Sẵn Sàng Trước Ngày Thi

- [ ] **A**: Toàn bộ dữ liệu thực tế đã được lọc nhiễu, gom cụm sự kiện và lập chỉ mục FAISS thành công, nạp vào bộ nhớ không xảy ra lỗi gián đoạn.
- [ ] **B**: AI Agent vận hành thông minh, xử lý mượt mà các câu hỏi đa lượt (KISC), định vị phân đoạn thời gian có độ chính xác cao trên ít nhất 20 câu query kiểm thử khó.
- [ ] **C**: Hệ thống API FastAPI vận hành mượt mà, chịu tải tốt, phân tách xử lý payload linh hoạt (Text/Image input), cấu hình kịch bản cứu hộ (fallback) sẵn sàng.
- [ ] **D**: Giao diện UI đáp ứng tiêu chuẩn phản hồi nhanh, hỗ trợ đầy đủ các tính năng tương tác vòng lặp (gợi ý concept thu hẹp/mở rộng tìm kiếm, gom cụm hiển thị video, phát đúng khung thời gian).
- [ ] **Cả đội**: Nắm vững chiến thuật đặt câu hỏi cốt lõi theo nguyên lý thu hẹp phễu thông tin: `Thời gian` $\rightarrow$ `Địa điểm` $\rightarrow$ `Đối tượng hành động` $\rightarrow$ `Cảnh vật/Vật thể chi tiết xung quanh`.

## 6. Hướng Dẫn Cài Đặt Môi Trường

### Yêu cầu
- Python 3.10+ đã được cài đặt
- Git

### Cài đặt nhanh (Windows)

**Bước 1**: Clone repo về máy

```bash
cd AI_Challenge_2026
```

**Bước 2**: Chạy script cài đặt tự động

```bash
setup.bat
```

Script sẽ tự động:
1. Tạo môi trường ảo `venv/`
2. Kích hoạt `venv`
3. Cài CLIP từ GitHub
4. Cài toàn bộ thư viện trong `requirements.txt`

**Bước 3**: Mỗi lần làm việc, kích hoạt lại venv:

```bash
venv\Scripts\activate.bat
```

### Lưu ý

> ⚠️ **torch / CUDA**: Nếu máy có GPU, cài `torch` theo đúng phiên bản CUDA tại [https://pytorch.org](https://pytorch.org) **trước** khi chạy `setup.bat`.

> ⚠️ **API Keys**: Tạo file `.env` ở thư mục gốc và điền các key cần thiết (không commit file này lên Git):
> ```
> OPENAI_API_KEY=sk-...
> ANTHROPIC_API_KEY=...
> GOOGLE_API_KEY=...
> ```
