# Bộ Test Truy Vấn AIC 2026 (>= 40 câu)

Dưới đây là bộ câu hỏi thử nghiệm đa dạng cho vòng Sơ tuyển, mô phỏng các dạng truy vấn thực tế. Đội cần chạy bộ test này trên tập dữ liệu thật và đo R-Score thông qua script `eval.py`.

## I. Dạng Textual KIS (Truy xuất từ mô tả văn bản thô) - 15 câu

Dạng này yêu cầu mô hình (CLIP) phải hiểu được các hành động, vật thể kết hợp, và màu sắc trong cùng một khung hình.

1. Một người đàn ông mặc áo khoác đỏ đang đạp xe trên con đường đất.
2. Hai chiếc ô tô đâm vào nhau tại một ngã tư đông đúc dưới trời mưa.
3. Một con chó màu vàng đang đuổi theo quả bóng trên thảm cỏ xanh.
4. Cảnh sát giao thông đang vung gậy điều tiết xe cộ vào ban đêm.
5. Một người phụ nữ đeo kính râm đang cầm cốc cà phê đi bộ ngang qua camera hành trình.
6. Một chiếc xe máy chở 3 người lạng lách qua đầu xe tải.
7. Vụ hỏa hoạn cháy lớn tại một khu chợ, khói bốc lên nghi ngút.
8. Người giao hàng đang để lại gói bưu kiện trước hiên nhà.
9. Đèn tín hiệu giao thông chuyển từ vàng sang đỏ và một chiếc sedan trắng vẫn cố vượt qua.
10. Góc quay camera an ninh: Một người đội mũ bảo hiểm đang cố bẻ khóa chiếc xe máy tay ga.
11. Đám đông khán giả đang cổ vũ nồng nhiệt trong một sân vận động khép kín.
12. Một vụ kẹt xe nghiêm trọng trên đường cao tốc với nhiều xe cứu thương.
13. Xe buýt màu xanh lá cây dừng lại ở trạm, có 2 người già bước xuống.
14. Chú mèo đen nhảy lên bàn và làm đổ cốc nước.
15. Khung cảnh ban đêm tĩnh lặng, chỉ có một chiếc xe đạp chạy lướt qua dưới ánh đèn đường.

## II. Dạng Q&A (Visual Question Answering) - 15 câu

Dạng này yêu cầu Agent / LLM kết hợp với hình ảnh lấy được từ FAISS để suy luận trả lời. Format nộp: `video_id`, `frame_id`, `answer`.

16. **Q:** Trong đoạn video có chiếc xe bồn chở xăng màu trắng đi ngang qua, biển số xe phía sau có chữ số mấy?
17. **Q:** Ở phân cảnh hai người đàn ông cãi nhau trước quán nhậu, người mặc áo đen đang cầm vật gì trên tay?
18. **Q:** Khi chiếc xe khách bị lật nghiêng, tên của hãng xe in trên hông xe là gì?
19. **Q:** Trong video camera an ninh ở tiệm vàng, kẻ gian mặc áo khoác màu gì khi đập vỡ tủ kính?
20. **Q:** Quán phở bên đường (nơi xảy ra vụ ẩu đả) có dòng chữ màu đỏ trên bảng hiệu viết gì?
21. **Q:** Có bao nhiêu người xuất hiện trong thang máy ngay trước khi cửa đóng lại?
22. **Q:** Người phụ nữ dắt chó đi dạo ở công viên đang đeo túi xách hãng nào?
23. **Q:** Trong vụ tai nạn xe máy ở vòng xuyến, nạn nhân đội mũ bảo hiểm màu gì?
24. **Q:** Chiếc xe tải vượt đèn đỏ thuộc công ty chuyển phát nào (dựa vào logo)?
25. **Q:** Ở giây thứ mấy thì ngọn lửa bắt đầu bùng lên từ nắp capo của chiếc xe con?
26. **Q:** Biển báo giao thông gần vụ tai nạn chỉ giới hạn tốc độ là bao nhiêu km/h?
27. **Q:** Đứa trẻ chạy ngang qua đường (suýt bị tông) đang mặc áo phông in hình con vật gì?
28. **Q:** Trong quán cafe, món đồ uống bị đổ lên bàn là màu gì?
29. **Q:** Chiếc xe cấp cứu xuất hiện trong khung hình chạy về hướng trái hay hướng phải của camera?
30. **Q:** Người đang leo rào vào nhà dân (camera hồng ngoại) có đeo khẩu trang hay không?

## III. Dạng TRAKE (Chuỗi sự kiện theo thời gian) - 10 câu

Dạng này yêu cầu module nâng cao của B (Agent + Temporal constraints) phải tìm đúng video chứa đủ chuỗi sự kiện, và báo cáo lại thứ tự các `frame_id` khớp với ngữ nghĩa.

31. **Nhảy qua suối:** Người đàn ông chạy lấy đà trên bờ (T1) -> Bật nhảy khỏi mặt đất (T2) -> Lơ lửng trên không trung (T3) -> Tiếp đất ở bờ bên kia (T4).
32. **Tai nạn giao thông:** Xe máy chạy tốc độ cao vượt phải (T1) -> Va chạm với sườn xe tải (T2) -> Ngã lăn ra đường (T3) -> Người dân xúm lại giúp đỡ (T4).
33. **Ném bóng rổ:** Cầu thủ dồi bóng (T1) -> Bật cao ném bóng (T2) -> Bóng bay vào rổ (T3) -> Khán giả ăn mừng (T4).
34. **Giao hàng:** Xe máy shipper đỗ lại (T1) -> Shipper cầm gói hàng đi bộ vào cửa (T2) -> Bấm chuông (T3) -> Chủ nhà mở cửa nhận hàng (T4).
35. **Kẻ trộm:** Một bóng đen nấp sau gốc cây (T1) -> Tiến lại gần chiếc xe đạp (T2) -> Dùng kìm cắt khóa (T3) -> Đạp xe phóng đi (T4).
36. **Cháy nhà:** Khói nhỏ bốc lên từ cửa sổ tầng 2 (T1) -> Lửa bùng phát lớn (T2) -> Xe cứu hỏa chạy đến (T3) -> Lính cứu hỏa đang xịt nước (T4).
37. **Giao thông:** Hai xe ô tô đối đầu nhau ở hẻm nhỏ (T1) -> Cả hai xe dừng lại (T2) -> Tài xế xe đỏ hạ kính mắng chửi (T3) -> Xe trắng từ từ lùi lại nhường đường (T4).
38. **Bắt cướp:** Tên cướp giật túi xách chạy bộ (T1) -> Cảnh sát đuổi theo phía sau (T2) -> Cảnh sát vật ngã tên cướp (T3) -> Còng tay tên cướp (T4).
39. **Sửa xe:** Lốp xe bị xẹp (T1) -> Người lái xe lấy kích nâng xe (T2) -> Tháo lốp cũ ra (T3) -> Lắp lốp dự phòng vào (T4).
40. **Bão lụt:** Cây cối nghiêng ngả vì gió lớn (T1) -> Nước dâng lên ngập bánh xe ô tô (T2) -> Một cành cây lớn gãy đổ đè lên nóc xe (T3) -> Đội cứu hộ bơi xuồng đến (T4).
