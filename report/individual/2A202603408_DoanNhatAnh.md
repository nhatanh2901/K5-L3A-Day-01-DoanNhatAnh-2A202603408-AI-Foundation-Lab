# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603408
- **Họ và tên:** Đoàn Nhật Anh
- **SEED** (in ở ô đầu notebook): 7901

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính
| Mô hình | Precision | Recall | F1 test | F1 SUB 40 | Train (s) | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoW + LogReg | 0.989 | 0.811 | 0.891 | 0.824 | 3.056 | 0.088 | 0.00 |
| BoW + MLP | 0.969 | 0.896 | 0.931 | 0.889 | 11.040 | 0.400 | 0.00 |
| MiniLM + kNN | 0.875 | 0.925 | 0.899 | 0.919 | 2.893 | 18.304 | 0.00 |
| MiniLM + LogReg | 0.960 | 0.906 | 0.932 | 0.919 | 3.394 | 7.108 | 0.00 |
| MiniLM + MLP head | 0.951 | 0.915 | 0.933 | 0.919 | 3.933 | 7.421 | 0.00 |
| LLM nhỏ Zero-shot | 0.944 | 0.850 | 0.895 | 0.895 | 0.000 | 1171.299 | 0.01 |
| LLM nhỏ Few-shot | 0.944 | 0.850 | 0.895 | 0.895 | 0.000 | 3001.923 | 0.01 |

## 3. Dự đoán và thực tế
1. Dự đoán về mô hình ML (ml_f1)

Tôi dự đoán F1 của mô hình ML tự cài đặt đạt khoảng 0.85, đồng thời precision cao hơn recall. Kết quả thực tế của BoW + Logistic Regression đạt F1 test = 0.891, precision = 0.989 và recall = 0.811. Như vậy, F1 cao hơn dự đoán ban đầu; precision cao nhưng recall thấp cho thấy mô hình phân loại spam khá chính xác khi đưa ra dự đoán spam, nhưng vẫn bỏ sót một số tin nhắn spam.

2. Dự đoán về Few-shot (llm_fewshot)

Tôi dự đoán Few-shot sẽ cải thiện F1 so với Zero-shot nhờ cung cấp các ví dụ có nhãn cho LLM. Tuy nhiên, kết quả thực tế cho thấy cả hai đều đạt F1 SUB 40 = 0.895. Việc thêm ví dụ chưa cải thiện chất lượng phân loại trong thí nghiệm này, trong khi độ trễ p95 của Few-shot tăng lên 3001.923 ms/tin so với 1171.299 ms/tin của Zero-shot.

3. Dự đoán về mô hình Embedding (emb_tricky)

Tôi dự đoán Encoder + Logistic Regression sẽ phân loại tốt hơn Bag-of-Words đối với những tin nhắn ngụy trang. Kết quả thực tế cho thấy MiniLM + Logistic Regression đạt F1 SUB 40 = 0.919, cao hơn BoW + Logistic Regression (0.824). Kết quả phù hợp với dự đoán vì Embedding biểu diễn ngữ nghĩa của nội dung, thay vì chỉ dựa vào sự xuất hiện của từ khóa.

## 4. Phân tích 50 case khó

BoW + Logistic Regression là mô hình giảm F1 nhiều nhất, từ 0.824 trên tập SUB xuống 0.333 trên 50 case khó, giảm 0.491. BoW + MLP cũng giảm từ 0.889 xuống 0.421.

Nhóm spam giọng tự nhiên là nhóm khó nhất đối với các mô hình truyền thống và Embedding. Trên nhóm này, BoW + Logistic Regression và BoW + MLP đều chỉ đạt accuracy 10%; các mô hình Embedding đạt khoảng 30–40%.

Hai trường hợp đáng chú ý:

- Tin nhắn thông báo trúng thưởng camera: Tin nhắn sử dụng cách diễn đạt giống một thông báo trao thưởng thông thường, nhưng thực chất có nội dung dụ người nhận thực hiện hành động. Mô hình dựa vào từ khóa có thể không nhận ra ý định lừa đảo nếu những từ đặc trưng của spam không xuất hiện rõ ràng.

- Tin nhắn quảng cáo cập nhật điện thoại miễn phí: Tin nhắn sử dụng ngôn ngữ tự nhiên, lời mời gọi hấp dẫn và thông tin liên hệ. Cách diễn đạt này khiến việc phân biệt quảng cáo lừa đảo với thông báo hợp lệ trở nên khó khăn hơn.

Trong thử nghiệm 50 case, LLM nhỏ Zero-shot đạt accuracy 98%, còn LLM kết hợp defensive prompt đạt 100%. Tuy nhiên, đây chỉ là kết quả trên tập kiểm tra được xây dựng riêng, chưa đủ để khẳng định mô hình sẽ đạt độ chính xác tương tự trong thực tế.

## 5. Thử thách cá nhân

Theo SEED = 7901, thử thách cá nhân của tôi là Challenge #6 – JSON output.

Tôi chỉnh sửa prompt để yêu cầu LLM trả về kết quả theo định dạng JSON gồm hai trường label và reason. Sau đó, tôi xây dựng phần phân tích JSON và chạy thử nghiệm trên 10 tin nhắn SUB.

Ở lần chạy đầu tiên, kết quả đạt F1 = 1.00 và Accuracy = 1.00, nhưng có 1 kết quả không hợp lệ. Tôi tiếp tục điều chỉnh prompt, yêu cầu mô hình chỉ trả về JSON hợp lệ, không thêm Markdown hoặc nội dung bên ngoài.

Ở lần chạy tiếp theo, kết quả đạt F1 = 1.00, Accuracy = 1.00 và Invalid = 0.

Qua thử thách này, tôi hiểu rằng prompt không chỉ ảnh hưởng đến nội dung dự đoán mà còn ảnh hưởng đến cấu trúc đầu ra. JSON giúp kết quả dễ kiểm tra và tích hợp vào chương trình, nhưng cần nhiều token đầu ra hơn so với việc chỉ trả về một nhãn.

## 6. Phương án cho sản phẩm thật

Với yêu cầu p95 dưới 50 ms/tin và xử lý 10 triệu SMS mỗi ngày, tôi lựa chọn MiniLM + Logistic Regression vì mô hình cân bằng tương đối tốt giữa chất lượng phân loại và tốc độ xử lý.

- F1 SUB 40 đạt 0.919.

- p95 đạt 7.108 ms/tin, thấp hơn giới hạn 50 ms.

- Chi phí API trong bảng thí nghiệm là 0 USD/1.000 tin do mô hình Embedding và Logistic Regression được chạy cục bộ.

So với BoW + Logistic Regression, MiniLM + Logistic Regression có F1 SUB cao hơn (0.919 so với 0.824), nhưng p95 cũng lớn hơn (7.108 ms so với 0.088 ms).

Tôi chấp nhận đánh đổi một phần tốc độ và tài nguyên tính toán để có khả năng phân loại tốt hơn. LLM không được chọn để xử lý toàn bộ tin nhắn vì p95 cao hơn nhiều so với giới hạn yêu cầu và có chi phí API.

Các số đo trên được thực hiện trong môi trường Colab. Khi triển khai thực tế, cần kiểm tra thêm tải đồng thời, tài nguyên máy chủ và khả năng đáp ứng 10 triệu tin nhắn mỗi ngày.

## 7. AI log và điều đã học

AI log:

- Hỏi về nguyên lý hoạt động của Logistic Regression, MLP, Embedding và LLM; nhận được giải thích về cách các mô hình thực hiện phân loại tin nhắn.

- Hỏi về các công thức sigmoid, cosine similarity, precision, recall và F1; nhận được hướng dẫn tính toán và giải thích ý nghĩa các chỉ số.

- Hỏi về cách xây dựng và đánh giá mô hình; nhận được hướng dẫn về cấu trúc mạng, quá trình huấn luyện và cách so sánh kết quả.

- Hỏi về prompt JSON và cách kiểm tra đầu ra; nhận được hướng dẫn về định dạng JSON và cách xử lý kết quả không hợp lệ. Tôi điều chỉnh prompt và kiểm tra lại bằng thực nghiệm.

Điều đã học: Qua bài lab, tôi hiểu rằng độ chính xác cao trên tập dữ liệu thông thường không đảm bảo mô hình xử lý tốt các trường hợp khó. Tôi cũng hiểu rõ hơn sự khác biệt giữa ML, DL, Embedding và LLM trong bài toán phân loại văn bản. Cuối cùng, khi lựa chọn mô hình cho sản phẩm thực tế, cần cân nhắc đồng thời độ chính xác, tốc độ xử lý và chi phí thay vì chỉ chọn mô hình có F1 cao nhất.
