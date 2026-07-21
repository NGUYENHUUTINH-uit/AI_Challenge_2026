# Bộ Câu Hỏi Luyện Tập Đồ Án AI Challenge 2026 (Người D thiết lập)

Bộ dữ liệu này bao gồm 30 câu hỏi mô phỏng cho bài toán KIS (Known-Item Search) và KISC (Known-Item Search with Conversation), dùng để đánh giá và tinh chỉnh mô hình retrieval.

## Phần 1: KIS (Known-Item Search) - 15 câu mô tả thô
Đây là các câu mô tả hình ảnh hoặc hành động thô bằng văn bản, thường dùng cho tìm kiếm một lượt (zero-shot/one-shot).

1. Một người đàn ông mặc áo đỏ đang chạy xe đạp trên đường đất.
2. Hai đứa trẻ đang chơi đùa với quả bóng màu xanh trên bãi cỏ.
3. Chiếc xe ô tô màu vàng đang vượt đèn đỏ tại ngã tư.
4. Một người phụ nữ đeo kính râm, đội nón lá đang cầm cốc cà phê.
5. Con chó mực đang sủa một người giao hàng mặc đồng phục màu cam.
6. Buổi hoàng hôn trên biển, có một chiếc thuyền buồm ở phía xa.
7. Đám đông đang chen lấn trong một buổi hòa nhạc ngoài trời.
8. Một vụ tai nạn giao thông giữa xe tải và xe máy dưới trời mưa.
9. Cảnh sát đang rượt đuổi một tên trộm mặc áo hoodie đen.
10. Một nhóm người đang nướng thịt BBQ tại công viên.
11. Bà lão đang ngồi đan len trước hiên nhà bằng gỗ.
12. Người đàn ông vừa gọi điện thoại vừa đi vội vã vào tòa nhà văn phòng.
13. Cận cảnh một con mèo đang vồ một con chuột đồ chơi.
14. Đội cứu hỏa đang phun nước vào một căn nhà đang bốc cháy.
15. Vận động viên điền kinh cán đích đầu tiên và giơ hai tay ăn mừng.

## Phần 2: KISC (Known-Item Search with Conversation) - 15 đoạn hội thoại đa lượt
Đây là các kịch bản người dùng tương tác với hệ thống qua nhiều lượt để làm rõ và tinh chỉnh kết quả tìm kiếm.

**Kịch bản 1: Tìm người**
- Lượt 1: Tìm cho tôi đoạn có một người đang đi bộ.
- Lượt 2: Không, người này mặc áo khoác màu xanh dương.
- Lượt 3: Người đó đang đi bộ dưới trời tuyết.

**Kịch bản 2: Tìm xe cộ**
- Lượt 1: Tôi muốn tìm cảnh một chiếc xe hơi đang chạy nhanh.
- Lượt 2: Đó là xe thể thao màu đỏ.
- Lượt 3: Xe đó đang chạy trên đường đua chuyên nghiệp, không phải đường phố.

**Kịch bản 3: Hoạt động thể thao**
- Lượt 1: Tìm video có người đang chơi thể thao.
- Lượt 2: Là môn bóng rổ, có người đang úp rổ.
- Lượt 3: Cầu thủ úp rổ mặc áo số 23.

**Kịch bản 4: Cảnh sinh hoạt**
- Lượt 1: Cảnh một bữa ăn gia đình.
- Lượt 2: Có một cậu bé đang làm đổ ly nước.
- Lượt 3: Mọi người đang ngồi trên chiếu, không phải bàn ăn.

**Kịch bản 5: Động vật**
- Lượt 1: Cảnh một đàn chim bay trên trời.
- Lượt 2: Đàn chim đang bay qua một hồ nước.
- Lượt 3: Là chim hồng hạc, màu hồng rất rõ.

**Kịch bản 6: Tai nạn**
- Lượt 1: Một vụ va chạm xe cộ.
- Lượt 2: Va chạm giữa xe buýt và xe đạp.
- Lượt 3: Xe buýt màu xanh lá cây và sự việc xảy ra ở bến xe.

**Kịch bản 7: Cảnh đêm**
- Lượt 1: Cảnh đường phố vào ban đêm.
- Lượt 2: Có rất nhiều đèn neon đủ màu sắc.
- Lượt 3: Trời đang mưa, mặt đường phản chiếu ánh đèn.

**Kịch bản 8: Đánh nhau**
- Lượt 1: Tìm đoạn có người đang xô xát.
- Lượt 2: Hai người phụ nữ đang cãi nhau trong siêu thị.
- Lượt 3: Một người đang ném đồ vật vào người kia.

**Kịch bản 9: Biểu diễn nghệ thuật**
- Lượt 1: Người đang hát trên sân khấu.
- Lượt 2: Ca sĩ là nam, đang đánh đàn guitar.
- Lượt 3: Sân khấu có khói mù mịt và ánh sáng màu tím.

**Kịch bản 10: Tự nhiên**
- Lượt 1: Cảnh thác nước.
- Lượt 2: Thác nước nằm trong rừng rậm.
- Lượt 3: Có người đang đứng trên mỏm đá nhìn xuống thác.

**Kịch bản 11: Làm việc**
- Lượt 1: Người đang làm việc trên máy tính.
- Lượt 2: Người đó có vẻ rất mệt mỏi, đang gục đầu xuống bàn.
- Lượt 3: Phòng làm việc tối om, chỉ có ánh sáng từ màn hình.

**Kịch bản 12: Nấu ăn**
- Lượt 1: Ai đó đang nấu ăn trong bếp.
- Lượt 2: Đang chiên xào món gì đó, lửa bốc lên rất cao (flambé).
- Lượt 3: Đầu bếp đội nón cao màu trắng.

**Kịch bản 13: Cảnh sát**
- Lượt 1: Xe cảnh sát đang làm nhiệm vụ.
- Lượt 2: Xe cảnh sát hú còi và nháy đèn xanh đỏ.
- Lượt 3: Xe đang dừng trước một cửa hàng trang sức bị đập vỡ kính.

**Kịch bản 14: Lễ hội**
- Lượt 1: Đám đông đang ăn mừng lễ hội.
- Lượt 2: Mọi người đang ném bột màu vào nhau.
- Lượt 3: Có một con voi được trang trí sặc sỡ đi giữa đám đông.

**Kịch bản 15: Sân bay**
- Lượt 1: Máy bay đang cất cánh.
- Lượt 2: Máy bay của hãng hàng không có logo màu đỏ.
- Lượt 3: Bầu trời lúc đó có rất nhiều mây đen sấm chớp.
