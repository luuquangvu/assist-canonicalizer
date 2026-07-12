# Assist Canonicalizer cho Home Assistant

[![Release](https://img.shields.io/github/v/release/luuquangvu/assist-canonicalizer?style=flat&logo=github&label=Release&color=purple)](https://github.com/luuquangvu/assist-canonicalizer/releases)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg?style=flat&logo=homeassistantcommunitystore&label=HACS)](https://github.com/hacs/integration)
[![Home Assistant](https://img.shields.io/badge/homeassistant-%3E%3D%202024.12.0-03a9f4?style=flat&logo=homeassistant&label=Home+Assistant)](https://www.home-assistant.io)

[![CI](https://img.shields.io/github/actions/workflow/status/luuquangvu/assist-canonicalizer/ci.yaml?style=flat&logo=github&label=CI)](https://github.com/luuquangvu/assist-canonicalizer/actions/workflows/ci.yaml)
[![Validation](https://img.shields.io/github/actions/workflow/status/luuquangvu/assist-canonicalizer/validation.yaml?style=flat&logo=github&label=Validation)](https://github.com/luuquangvu/assist-canonicalizer/actions/workflows/validation.yaml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/luuquangvu/assist-canonicalizer/github-code-scanning/codeql?style=flat&logo=github&label=CodeQL)](https://github.com/luuquangvu/assist-canonicalizer/actions/workflows/github-code-scanning/codeql)
[![Prettier](https://img.shields.io/github/actions/workflow/status/luuquangvu/assist-canonicalizer/prettier.yaml?style=flat&logo=prettier&label=Prettier)](https://github.com/luuquangvu/assist-canonicalizer/actions/workflows/prettier.yaml)

**[ [🇺🇸 English](README.md) | 🇻🇳 Tiếng Việt ]**

**Assist Canonicalizer** nâng cao độ chính xác khi nhận diện ý định (intent) của Home Assistant Assist thông qua việc chuẩn hóa các câu lệnh ngôn ngữ tự nhiên bằng công cụ xếp hạng từ vựng đa tín hiệu (multi-signal lexical ranking engine) trước khi chuyển tới tác nhân hội thoại tích hợp. Bộ tích hợp hoạt động như một tác nhân hội thoại tích hợp sâu (native conversation agent), giúp giảm thiểu tối đa tỷ lệ nhận diện sai và tỷ lệ chuyển tiếp dự phòng (fallback), đồng thời chạy hoàn toàn cục bộ trên thiết bị của bạn, không cần LLM, không đám mây, không thu thập dữ liệu (telemetry) và không gọi dịch vụ bên ngoài.

---

## Mục lục

- [Assist Canonicalizer cho Home Assistant](#assist-canonicalizer-cho-home-assistant)
  - [Mục lục](#mục-lục)
  - [Tính năng nổi bật](#tính-năng-nổi-bật)
  - [Cài đặt](#cài-đặt)
    - [Cách 1: Sử dụng HACS (Khuyên dùng)](#cách-1-sử-dụng-hacs-khuyên-dùng)
    - [Cách 2: Cài đặt thủ công](#cách-2-cài-đặt-thủ-công)
  - [Hướng dẫn thiết lập & Cấu hình](#hướng-dẫn-thiết-lập--cấu-hình)
  - [Nguyên lý hoạt động](#nguyên-lý-hoạt-động)
  - [Hiệu năng đo kiểm (Benchmark Performance)](#hiệu-năng-đo-kiểm-benchmark-performance)
  - [Các hành động trong Công cụ nhà phát triển](#các-hành-động-trong-công-cụ-nhà-phát-triển)
    - [Thử nghiệm so khớp (Test Match)](#thử-nghiệm-so-khớp-test-match)
    - [Xây dựng lại chỉ mục (Rebuild Index)](#xây-dựng-lại-chỉ-mục-rebuild-index)
    - [Xóa chỉ mục (Clear Index)](#xóa-chỉ-mục-clear-index)
    - [Chẩn đoán (Diagnostics)](#chẩn-đoán-diagnostics)
    - [Xuất danh sách ứng cử viên (Dump Candidates)](#xuất-danh-sách-ứng-cử-viên-dump-candidates)
  - [Kiểm soát độ tin cậy & Chuyển tiếp dự phòng (Confidence Gates & Fallback)](#kiểm-soát-độ-tin-cậy--chuyển-tiếp-dự-phòng-confidence-gates--fallback)
  - [Khắc phục sự cố & Gỡ lỗi](#khắc-phục-sự-cố--gỡ-lỗi)
    - [Các sự cố thường gặp](#các-sự-cố-thường-gặp)
    - [Quy trình chẩn đoán lỗi](#quy-trình-chẩn-đoán-lỗi)
  - [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
  - [Chất lượng Mã nguồn & Bảo mật](#chất-lượng-mã-nguồn--bảo-mật)
  - [Đóng góp](#đóng-góp)
  - [Bản quyền](#bản-quyền)
  - [Hỗ trợ dự án](#hỗ-trợ-dự-án)

---

## Tính năng nổi bật

- **Tác nhân hội thoại tích hợp sâu (Native Conversation Agent)**: Hoạt động như một tác nhân hội thoại chính thức trong Home Assistant. Tự động tiếp nhận mọi câu lệnh Assist, tiền xử lý thông qua cơ chế chuẩn hóa từ vựng (lexical canonicalization), rồi chuyển tiếp kết quả đã xác thực đến luồng xử lý hội thoại hiện tại của bạn một cách mượt mà.
- **Bộ chấm điểm xếp hạng từ vựng đa tín hiệu (Multi-Signal Lexical Ranking Engine)**: Đánh giá và chấm điểm từng câu lệnh ứng cử viên (candidate) dựa trên sự kết hợp của 4 thuật toán bổ trợ: **so khớp mờ RapidFuzz**, **độ tương đồng n-gram Jaccard**, **truy xuất xác suất BM25**, và **so khớp hành động theo miền ý định (intent domain action matching)**. Cơ chế chấm điểm có trọng số này mang lại độ chính xác vượt trội so với việc chỉ sử dụng một thuật toán đơn lẻ.
- **Tự động xây dựng chỉ mục câu lệnh**: Tự động tạo và cập nhật cơ sở dữ liệu mẫu câu chuẩn hóa từ mọi nguồn hiện có: ý định (intent) mặc định của Home Assistant, các tệp YAML chứa mẫu câu tùy chỉnh, tên thực thể và bí danh (alias), sơ đồ phòng/tầng (area/floor) và các tham số (slot) mở rộng động. Mọi quá trình đều diễn ra tự động mà không cần cấu hình thủ công.
- **Tối ưu tốc độ tải bằng bộ nhớ đệm**: Lưu trữ chỉ mục câu lệnh đã chuẩn hóa trực tiếp vào phân lớp lưu trữ của Home Assistant. Nhờ vậy, hệ thống có thể bỏ qua bước phân tích cú pháp (parsing) mẫu câu ở các lần khởi động tiếp theo, tối ưu đáng kể thời gian tải.
- **Bộ lọc độ tin cậy linh hoạt**: Cho phép tinh chỉnh hành vi so khớp qua hai ngưỡng cấu hình: **Độ tin cậy tối thiểu (Minimum Match Confidence)** và **Khoảng cách độ tin cậy (Minimum Confidence Margin)**. Chỉ những câu lệnh đạt đủ điều kiện của cả hai ngưỡng mới được phê duyệt chuẩn hóa và thực thi.
- **Cơ chế chuyển tiếp dự phòng đa cấp an toàn**: Trong trường hợp chỉ mục trống, độ tin cậy xếp hạng thấp, hoặc xảy ra xung đột ý định (hai intent khác nhau có điểm số tương đương), hệ thống sẽ tự động chuyển tiếp câu lệnh gốc của bạn tới một tác nhân hội thoại dự phòng (fallback agent) - ví dụ như một mô hình ngôn ngữ lớn (LLM) để xử lý tiếp, đảm bảo trải nghiệm sử dụng không bị gián đoạn.
- **Bộ công cụ chẩn đoán chuyên sâu cho nhà phát triển**: Cung cấp 5 hành động (service actions) chuyên dụng bao gồm `test_match`, `rebuild_index`, `clear_index`, `diagnostics`, và `dump_candidates` giúp bạn dễ dàng theo dõi quá trình xếp hạng, kiểm tra dữ liệu chỉ mục và quản lý vòng đời bộ nhớ đệm ngay trên giao diện Home Assistant.
- **Phân tách cơ sở dữ liệu theo ngôn ngữ**: Quản lý độc lập chỉ mục câu lệnh cho từng ngôn ngữ khác nhau, hỗ trợ tự động nhận diện và ánh xạ theo các biến thể ngôn ngữ của Home Assistant.
- **Kiểm soát và giới hạn tài nguyên hệ thống**: Tích hợp sẵn cơ chế giới hạn số lượng mẫu câu trên mỗi ngôn ngữ, intent hoặc mỗi lượt xếp hạng nhằm ngăn ngừa tối đa nguy cơ quá tải bộ nhớ (out-of-memory).
- **Hoạt động ngoại tuyến 100% (Local-only)**: Đạt tiêu chuẩn phân loại thiết bị `local_push`. Mọi tác vụ từ tiền xử lý, lập chỉ mục, chấm điểm cho đến xác thực đều được thực hiện trực tiếp trên phần cứng Home Assistant của bạn. Không phụ thuộc đám mây, không gửi dữ liệu ra ngoài và không cần kết nối internet.

---

## Cài đặt

### Cách 1: Sử dụng HACS (Khuyên dùng)

[![Add Assist Canonicalizer to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=luuquangvu&repository=assist-canonicalizer&category=integration)

1. Mở **HACS** trong Home Assistant.
2. Tìm kiếm **Assist Canonicalizer**.
3. Nếu không tìm thấy, nhấp vào biểu tượng ba chấm ở góc trên cùng bên phải và chọn **Kho lưu trữ tùy chỉnh (Custom repositories)**.
4. Thêm `https://github.com/luuquangvu/assist-canonicalizer` với danh mục **Bộ tích hợp (Integration)**.
5. Tìm kiếm **Assist Canonicalizer** và nhấp vào **Tải xuống (Download)**.
6. Khởi động lại Home Assistant.

### Cách 2: Cài đặt thủ công

1. Tải bản phát hành mới nhất và giải nén các tệp tin.
2. Sao chép thư mục `custom_components/assist_canonicalizer` vào thư mục `config/custom_components/` của Home Assistant.
3. Khởi động lại Home Assistant.

---

## Hướng dẫn thiết lập & Cấu hình

1. Vào **Cài đặt (Settings)** > **Thiết bị & Dịch vụ (Devices & Services)**.
2. Chọn **Thêm tích hợp (Add Integration)** và tìm kiếm **Assist Canonicalizer**.
3. Chọn **Tác nhân hội thoại dự phòng (Fallback Conversation Agent)**: Khi Assist Canonicalizer không thể tìm thấy câu lệnh phù hợp với độ tin cậy cao, câu lệnh gốc của bạn sẽ được chuyển tiếp tới tác nhân dự phòng này. Bạn không nên chọn tác nhân mặc định của Home Assistant làm dự phòng, vì bản thân câu chuẩn hóa đã được xác thực qua nó và thất bại; việc gửi lại câu lệnh thô thường cũng sẽ thất bại. Hãy chọn một tác nhân hội thoại sử dụng LLM để có khả năng diễn giải ngữ cảnh linh hoạt nhất.
4. Cấu hình **Độ tin cậy tối thiểu (Minimum Match Confidence)**: Yêu cầu câu lệnh ứng cử viên phải đạt điểm số bằng hoặc cao hơn ngưỡng này trên cả 4 thuật toán xếp hạng mới được chấp nhận. Khuyên dùng giữ nguyên cấu hình mặc định ban đầu và sử dụng công cụ **Test Match** để theo dõi điểm số thực tế trước khi tùy biến.
5. Cấu hình **Khoảng cách độ tin cậy tối thiểu (Minimum Confidence Margin)**: Khoảng cách điểm số tối thiểu giữa ứng cử viên xếp hạng cao nhất và ứng cử viên tiếp theo (thuộc intent khác). Ngưỡng này giúp loại bỏ sự mơ hồ khi câu lệnh đầu vào khớp với nhiều ý định khác nhau có điểm số xấp xỉ nhau.
6. Vào **Cài đặt (Settings)** > **Trợ lý giọng nói (Voice assistants)** và mở Assist pipeline của bạn. Tại mục **Tác nhân hội thoại (Conversation agents)**, chọn **Assist Canonicalizer** từ danh sách tác nhân.

> [!IMPORTANT]
> **Lưu ý quan trọng**: Assist Canonicalizer chỉ thực sự hoạt động sau khi được cấu hình và kích hoạt trong Assist pipeline của bạn.
>
> Hãy bắt đầu với các giá trị ngưỡng mặc định và điều chỉnh dần trong quá trình sử dụng. Nếu bộ chuẩn hóa chuyển tiếp dự phòng quá thường xuyên, hãy thử hạ thấp `min_confidence`. Nếu nó khớp sai intent, hãy tăng `min_confidence` và `min_margin`.

---

## Nguyên lý hoạt động

Khi bạn đọc câu lệnh (qua STT) hoặc gửi tin nhắn trong ô chat Assist, Assist Canonicalizer sẽ tiếp quản câu lệnh đó và xử lý theo quy trình dưới đây:

```mermaid
flowchart TD
    A[Nhập liệu từ người dùng] --> B[Chuẩn hóa văn bản]
    B --> C[Tra cứu chỉ mục]
    C --> D[Xếp hạng đa tín hiệu]
    D --> E{Đạt ngưỡng tin cậy?}
    E -->|Đạt| F[Xác thực qua HA Agent]
    E -->|Không đạt| G[Tác nhân dự phòng]
    F -->|Được chấp nhận| H[Kết quả chuẩn hóa]
    F -->|Bị từ chối| G
    G --> I[Chuyển tiếp câu lệnh gốc]
```

1. **Chuẩn hóa văn bản (Text Normalization)**: Loại bỏ dấu câu, đưa văn bản về chữ thường, loại bỏ các khoảng trắng thừa và chuẩn hóa ký tự theo định dạng NFKC. Quy trình này được áp dụng đồng bộ cho cả câu lệnh đầu vào lẫn các mẫu câu ứng cử viên để đảm bảo tính nhất quán khi so sánh.

2. **Tra cứu chỉ mục (Index Lookup)**: Câu lệnh đã chuẩn hóa sẽ được tra cứu trong cơ sở dữ liệu mẫu câu đã được xây dựng sẵn cho ngôn ngữ hiện tại, bao gồm:
   - **Intent mặc định**: Toàn bộ câu lệnh cố định và mở rộng template từ cấu hình ngôn ngữ gốc của Home Assistant.
   - **Câu lệnh tùy chỉnh**: Định nghĩa trong các tệp YAML `custom_sentences/<lang>/`, intent script trong `configuration.yaml` hoặc các automation câu lệnh được tạo từ giao diện (UI).
   - **Thực thể (Entities)**: Tên và bí danh (alias) của các thực thể được chia sẻ (exposed entities).
   - **Khu vực & Tầng (Areas & Floors)**: Sơ đồ phòng và tầng trong ngôi nhà.

3. **Chấm điểm và xếp hạng (Multi-Signal Ranking)**: Từng mẫu câu phù hợp sẽ được đánh giá độc lập qua 4 tiêu chí trước khi tính điểm số tổng hợp cuối cùng:
   - **Độ tương đồng từ (Word Similarity)**: Đo lường độ khớp của các từ và thứ tự xuất hiện, hỗ trợ xử lý lỗi gõ sai chính tả hoặc tráo đổi vị trí từ.
   - **So khớp mẫu ký tự (Character Pattern Matching)**: Phân tích các phân đoạn ký tự (character n-grams) chồng chéo để nhận diện các từ có cách viết tương đồng.
   - **Độ liên quan từ khóa (Keyword Relevance)**: Thuật toán BM25 đánh giá mức độ quan trọng và tính đặc trưng của từng từ trong câu lệnh của bạn.
   - **Ngữ cảnh ý định (Intent Context)**: Ưu tiên các kết quả có loại ý định (ví dụ: bật đèn, điều chỉnh nhiệt độ) đồng nhất với các kết quả khớp tốt nhất, nhằm giảm thiểu sai lệch ngữ cảnh.

4. **Kiểm tra ngưỡng độ tin cậy (Confidence Gate)**: Hệ thống đánh giá xem câu lệnh ứng cử viên đứng đầu có vượt qua cả hai ngưỡng độ tin cậy `min_confidence` và `min_margin` đã cấu hình hay không. Nếu không đạt yêu cầu, cơ chế dự phòng sẽ được kích hoạt.

5. **Xác thực câu lệnh (Validation)**: Câu lệnh chuẩn hóa tối ưu nhất sẽ được gửi thử nghiệm tới bộ xử lý hội thoại mặc định (HassIL). Nếu bộ xử lý chấp nhận, lệnh sẽ được thực thi; ngược lại, hệ thống sẽ chuyển tiếp sang tác nhân dự phòng.

6. **Chuyển tiếp dự phòng (Fallback)**: Khi việc tìm kiếm so khớp hoặc xác thực thất bại, câu lệnh thô gốc ban đầu của bạn sẽ được chuyển tiếp thẳng tới tác nhân dự phòng để tiếp tục xử lý, đảm bảo trợ lý ảo vẫn phản hồi bình thường.

---

## Hiệu năng đo kiểm (Benchmark Performance)

Công cụ xếp hạng được đo kiểm (benchmark) bằng cách sử dụng các bộ dữ liệu thử nghiệm thực tế trên 5 ngôn ngữ (DE, EN, FR, NL, VI).

### Kết quả tổng quan

<!-- BENCHMARK_OVERALL_START -->

> Phiên bản phụ thuộc benchmark: `homeassistant` 2026.7.2, `home-assistant-intents` 2026.6.24.

| Chế độ    | Đúng Intent/Slot | Nhận diện sai (Mismatch) | Dự phòng (Fallback) |
| :-------- | ---------------: | -----------------------: | ------------------: |
| `hassil`  |            47.7% |                     0.0% |               52.3% |
| `lexical` |        **92.7%** |                 **1.5%** |            **5.8%** |

> Độ chính xác nhận diện Intent/Slot tăng từ **47.7% lên 92.7%**. Tổng tỷ lệ lỗi (nhận diện sai + chuyển sang dự phòng) giảm mạnh từ **52.3% xuống còn 7.3%**.

<!-- BENCHMARK_OVERALL_END -->

### Chi tiết theo từng ngôn ngữ

<!-- BENCHMARK_LANGS_START -->

| Ngôn ngữ | Chế độ    | Đúng Intent/Slot | Nhận diện sai (Mismatch) | Dự phòng (Fallback) |
| :------- | :-------- | ---------------: | -----------------------: | ------------------: |
| EN       | `hassil`  |            52.7% |                     0.0% |               47.3% |
| EN       | `lexical` |        **92.2%** |                 **2.3%** |            **5.4%** |
| DE       | `hassil`  |            48.4% |                     0.0% |               51.6% |
| DE       | `lexical` |        **95.1%** |                 **0.8%** |            **4.1%** |
| FR       | `hassil`  |            49.6% |                     0.0% |               50.4% |
| FR       | `lexical` |        **93.3%** |                 **1.7%** |            **5.0%** |
| NL       | `hassil`  |            48.8% |                     0.0% |               51.2% |
| NL       | `lexical` |        **89.9%** |                 **1.6%** |            **8.5%** |
| VI       | `hassil`  |            37.0% |                     0.0% |               63.0% |
| VI       | `lexical` |        **93.0%** |                 **1.0%** |            **6.0%** |

<!-- BENCHMARK_LANGS_END -->

> [!NOTE]
> Trong bài kiểm tra hiệu năng này, chúng ta giả định rằng khoảng một nửa số lần các ý định của chúng ta hoạt động với HassIL mặc định. Thực tế có thể khác do thói quen sử dụng của bạn.
>
> Độ chính xác, số lượng ứng viên và độ trễ có thể thay đổi tùy thuộc vào môi trường, bao gồm các yếu tố như số lượng thực thể, khu vực, tầng, phiên bản Home Assistant và phần cứng được sử dụng.

Tất cả kết quả đo kiểm trên được tạo ra bằng các bộ dữ liệu kiểm thử thực tế tại thư mục [`tests/real_world/`](tests/real_world/). Báo cáo chi tiết có tại thư mục [`benchmark/`](benchmark/). Kết quả thực tế trên hệ thống của bạn sẽ khác biệt tùy thuộc cấu hình thiết bị Home Assistant cụ thể.

---

## Các hành động trong Công cụ nhà phát triển

Tất cả các hành động của bộ tích hợp đều có thể truy cập từ **Công cụ nhà phát triển** > **Hành động** (Developer Tools > Actions) trong Home Assistant.

### Thử nghiệm so khớp (Test Match)

**Hành động**: `assist_canonicalizer.test_match`

Chạy thử nghiệm toàn bộ luồng xử lý và trả về kết quả chấm điểm chi tiết của các ứng cử viên. Công cụ này rất hữu ích để phân tích lý do câu lệnh được chấp nhận hoặc bị chuyển tiếp dự phòng, từ đó điều chỉnh các ngưỡng lọc cho phù hợp.

| Trường     | Bắt buộc | Mô tả                                        |
| ---------- | -------- | -------------------------------------------- |
| `text`     | Có       | Câu lệnh đầu vào cần thử nghiệm              |
| `language` | Không    | Mã ngôn ngữ (tự động nhận diện nếu bỏ trống) |

**Kết quả trả về bao gồm**:

- `normalized_text`: Dạng câu lệnh sau khi chuẩn hóa.
- `top_candidates`: Danh sách ứng cử viên điểm số cao nhất kèm chi tiết điểm từ các thuật toán (`rapidfuzz`, `char_ngram`, `bm25`, `intent`, điểm số cuối cùng `final`).
- `selected_candidate`: Câu lệnh được chọn và ý định (intent) tương ứng.
- `accepted`: Trạng thái duyệt (đã vượt qua các ngưỡng lọc hay chưa).

### Xây dựng lại chỉ mục (Rebuild Index)

**Hành động**: `assist_canonicalizer.rebuild_index`

Kích hoạt thủ công quá trình quét dữ liệu và xây dựng lại chỉ mục câu lệnh cho một ngôn ngữ nhất định. Hành động này tích hợp cơ chế chống trùng lặp (deduplicated): nếu tiến trình xây dựng chỉ mục đang chạy cho ngôn ngữ đó, yêu cầu mới sẽ được gộp chung và đợi tiến trình hiện tại hoàn thành.

| Trường     | Bắt buộc | Mô tả                                        |
| ---------- | -------- | -------------------------------------------- |
| `language` | Không    | Mã ngôn ngữ cần xây dựng lại dữ liệu chỉ mục |

### Xóa chỉ mục (Clear Index)

**Hành động**: `assist_canonicalizer.clear_index`

Xóa bộ nhớ đệm chỉ mục của một ngôn ngữ cụ thể (hoặc tất cả các ngôn ngữ nếu để trống). Hành động này cũng sẽ xóa sạch dữ liệu chỉ mục được lưu trữ trên đĩa cứng của Home Assistant.

| Trường     | Bắt buộc | Mô tả                             |
| ---------- | -------- | --------------------------------- |
| `language` | Không    | Mã ngôn ngữ cần xóa chỉ mục cache |

### Chẩn đoán (Diagnostics)

**Hành động**: `assist_canonicalizer.diagnostics`

Trả về thông tin trạng thái hoạt động theo thời gian thực của bộ tích hợp, bao gồm:

- `candidate_count`: Số lượng mẫu câu trong chỉ mục đang hoạt động.
- `index_version`: Phiên bản của chỉ mục hiện hành.
- `last_query_latency_ms`: Thời gian xử lý của câu lệnh gần nhất (ms).
- `last_fallback_reason`: Lý do câu lệnh gần nhất bị chuyển tiếp dự phòng.
- `last_error`: Lỗi gần nhất ghi nhận được.
- `dynamic_candidate_count`: Số lượng ứng cử viên được sinh ra tự động từ các thực thể hệ thống.
- `cached_languages`: Danh sách ngôn ngữ có dữ liệu chỉ mục lưu trên RAM.
- `cached_candidate_counts`: Số lượng ứng cử viên của từng ngôn ngữ trên RAM.
- `pending_rebuild_languages`: Các ngôn ngữ đang trong hàng đợi xây dựng lại chỉ mục.
- `registry_slot_counts`: Số lượng giá trị đăng ký theo từng thành phần (tên thực thể, phòng,...).
- `dynamic_candidate_generation`: Trạng thái và giới hạn tài nguyên của tiến trình mở rộng ứng cử viên động.
- `subscribed_intent_source_counts`: Số lượng intent đăng ký nhận dữ liệu từ các tác nhân khác.

### Xuất danh sách ứng cử viên (Dump Candidates)

**Hành động**: `assist_canonicalizer.dump_candidates`

Trích xuất thông tin chi tiết về cơ cấu chỉ mục của một ngôn ngữ, bao gồm số lượng câu từ các nguồn, thống kê theo intent, số lượng giá trị thực thể và danh sách mẫu câu cụ thể. Hữu ích để kiểm tra xem một câu lệnh đã được đưa vào cơ sở dữ liệu của bộ chuẩn hóa hay chưa.

| Trường     | Bắt buộc | Mô tả                                                                            |
| ---------- | -------- | -------------------------------------------------------------------------------- |
| `language` | Không    | Mã ngôn ngữ cần kiểm tra                                                         |
| `rebuild`  | Không    | Nếu là `true`, sẽ làm mới dữ liệu ứng cử viên trước khi xuất (mặc định: `false`) |

---

## Kiểm soát độ tin cậy & Chuyển tiếp dự phòng (Confidence Gates & Fallback)

Bộ tích hợp sử dụng hai ngưỡng cấu hình để phê duyệt xem có áp dụng câu lệnh ứng cử viên có điểm số cao nhất hay không:

**Độ tin cậy tối thiểu (Minimum Match Confidence - `min_confidence`)**
: Điểm số tổng hợp có trọng số của ứng cử viên dẫn đầu phải lớn hơn hoặc bằng giá trị này. Điểm số dao động từ `0.0` (hoàn toàn không khớp) đến `1.0` (khớp tuyệt đối trên cả 4 thuật toán).

**Khoảng cách độ tin cậy tối thiểu (Minimum Confidence Margin - `min_margin`)**
: Khoảng cách điểm số bắt buộc giữa ứng cử viên dẫn đầu và ứng cử viên tiếp theo có intent khác biệt. Điều này giúp hệ thống chủ động từ chối xử lý khi câu lệnh mơ hồ khiến nhiều ý định khác nhau có điểm số gần tương đương nhau.

Khi câu lệnh bị **chuyển tiếp dự phòng (fallback)**, nguyên nhân cụ thể sẽ được lưu lại trong bảng chẩn đoán dưới các mã sau:

| Lý do                  | Ý nghĩa                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `low_confidence`       | Không có câu lệnh ứng cử viên nào đạt ngưỡng điểm `min_confidence`.                                              |
| `low_margin`           | Điểm số của ứng cử viên đứng đầu và ứng cử viên tiếp theo (intent khác) quá sát nhau (dưới ngưỡng `min_margin`). |
| `empty_index`          | Chỉ mục câu lệnh của ngôn ngữ hiện tại chưa được xây dựng.                                                       |
| `validation_failed`    | Câu lệnh chuẩn hóa được chấp nhận nhưng bị bộ nhận diện mặc định của Home Assistant từ chối khi chạy thử nghiệm. |
| `ranking_failed`       | Xảy ra lỗi xử lý trong bước chấm điểm và xếp hạng.                                                               |
| `unexpected_exception` | Gặp lỗi nghiêm trọng không xác định trong quá trình thực thi.                                                    |

Bạn có thể kiểm tra lý do dự phòng của câu lệnh gần nhất bằng cách chạy hành động **Diagnostics**.

---

## Khắc phục sự cố & Gỡ lỗi

### Các sự cố thường gặp

**Bộ chuẩn hóa luôn chuyển sang dự phòng và không bao giờ so khớp thành công.**

1. Hãy sử dụng hành động **Diagnostics** để kiểm tra `last_fallback_reason`. Nếu lý do là `empty_index`, có nghĩa là chỉ mục chưa được tạo. Chỉ mục sẽ được chủ động làm nóng (warmup) ngay khi khởi động/tải lại cho tất cả ngôn ngữ được cấu hình trong Assist pipeline. Nếu `empty_index` vẫn xuất hiện, quá trình warmup có thể đã bị bỏ qua (không có pipeline nào được cấu hình, không có ngôn ngữ mặc định), hoặc quá trình xây dựng trong nền chưa hoàn tất. Bạn có thể kích hoạt thủ công qua hành động **Rebuild Index**.
2. Nếu lý do là `low_confidence`, ngưỡng `min_confidence` bạn đặt có thể quá cao. Hãy thử hạ thấp cấu hình này xuống. Bạn nên dùng công cụ **Test Match** để xem điểm số thực tế của các câu lệnh mẫu.
3. Nếu lý do là `validation_failed`, câu lệnh đã được so khớp thành công nhưng bị tác nhân mặc định của Home Assistant (HassIL) từ chối khi thực thi. Hãy chạy hành động **Dump Candidates** để kiểm tra xem cấu trúc câu lệnh chuẩn hóa của bạn đã chính xác và khớp với các định nghĩa intent chưa.

**Mẫu câu tùy chỉnh của tôi không được nhận diện.**

1. Kiểm tra lại cấu hình mẫu câu tùy chỉnh của bạn (nằm trong thư mục `config/custom_sentences/<lang>/`, intent script trong `configuration.yaml` hoặc automation từ giao diện). Đảm bảo mã ngôn ngữ được khai báo chính xác.
2. Chạy hành động **Dump Candidates** với tham số `rebuild: true` cho ngôn ngữ của bạn. Kiểm tra phần thống kê nguồn (`source` counts): nếu chỉ số `custom_sentence` bằng 0, có nghĩa là các tệp YAML của bạn chưa được Home Assistant tải lên thành công.
3. Chắc chắn các tệp YAML tuân thủ đúng [cú pháp mẫu câu của Home Assistant](https://www.home-assistant.io/voice_control/custom_sentences/).
4. Chạy hành động **Rebuild Index** để làm mới lại cơ sở dữ liệu sau khi sửa đổi các tệp cấu hình mẫu câu.

**Bộ tích hợp xử lý chậm ở câu lệnh đầu tiên.**

Chỉ mục cho các ngôn ngữ đã cấu hình trong Assist pipeline được xây dựng chủ động trong nền ngay khi khởi động và sau khi tải lại. Trong điều kiện bình thường, câu lệnh đầu tiên cho một ngôn ngữ sẽ truy cập vào bộ nhớ đệm đã sẵn sàng và không gặp độ trễ khởi tạo.

Nếu vẫn có độ trễ, chỉ mục có thể chưa hoàn tất quá trình xây dựng (kiểm tra `pending_rebuild_languages` trong kết quả **Diagnostics**). Các ngôn ngữ nằm ngoài cấu hình pipeline của bạn sẽ được xây dựng lười (lazy) ở lần sử dụng đầu tiên. Các lượt truy vấn tiếp theo sẽ sử dụng chỉ mục trong RAM nên tốc độ xử lý sẽ cực kỳ nhanh.

**Tôi mới cập nhật thực thể/khu vực/tầng nhưng bộ chuẩn hóa chưa nhận diện.**

Assist Canonicalizer tự động lắng nghe các sự kiện thay đổi từ sổ đăng ký thực thể, khu vực, tầng của Home Assistant và sẽ tự động xây dựng lại chỉ mục sau 5 giây (debounce). Nếu thay đổi chưa có hiệu lực ngay lập tức, hãy đợi một vài giây hoặc chạy hành động **Rebuild Index** để đồng bộ tức thì.

### Quy trình chẩn đoán lỗi

Để khắc phục sự cố một cách bài bản, hãy thực hiện theo các bước sau:

1. **Kiểm tra trạng thái chung**: Chạy hành động **Diagnostics** để xem tổng số mẫu câu trong chỉ mục, phiên bản dữ liệu hiện tại, độ trễ xử lý và lý do dự phòng của câu lệnh gần nhất.
2. **Kiểm tra cơ sở dữ liệu**: Chạy **Dump Candidates** với tùy chọn `rebuild: true` để xem chi tiết cơ cấu mẫu câu theo nguồn, mức độ phủ của các intent và danh sách các câu mẫu.
3. **Phân tích câu lệnh lỗi**: Chạy hành động **Test Match** với chính xác câu lệnh gặp sự cố. Quan sát mảng dữ liệu `top_candidates` để phân tích điểm số chi tiết từ các thuật toán (`rapidfuzz_score`, `char_ngram_score`, `bm25_score`, `intent_score`, `final_score`) nhằm hiểu rõ vì sao mẫu câu đó không vượt qua được các ngưỡng lọc.
4. **So sánh với tác nhân dự phòng**: Nếu câu lệnh bị chuyển dự phòng, hãy so sánh kết quả xử lý của bộ chuẩn hóa và tác nhân dự phòng qua công cụ **Test Match**. Nếu điểm số của ứng cử viên phù hợp chỉ thấp hơn ngưỡng cấu hình một chút, bạn có thể cân nhắc hạ nhẹ ngưỡng tin cậy. Nếu tác nhân dự phòng cũng xử lý kém, bạn cần bổ sung thêm mẫu câu phù hợp hơn trong tệp cấu hình.
5. **Tinh chỉnh các ngưỡng lọc**: Dựa trên điểm số phân tích từ **Test Match**, điều chỉnh các tham số `min_confidence` và `min_margin` trong phần cấu hình tích hợp, sau đó chạy lại **Test Match** để kiểm chứng.
6. **Xem nhật ký hệ thống (Logs)**: Tra cứu các dòng log của Home Assistant liên quan đến miền `assist_canonicalizer` để tìm kiếm thêm thông tin chi tiết lỗi.

---

## Yêu cầu hệ thống

- **Home Assistant** `>= 2024.12.0`
- Bộ tích hợp yêu cầu miền `conversation` và phụ thuộc vào sự sẵn có của thành phần `assist_pipeline`. Hoạt động tốt với mọi tác nhân hội thoại được Home Assistant hỗ trợ.

---

## Chất lượng Mã nguồn & Bảo mật

Để duy trì tiêu chuẩn cao về độ tin cậy và sự ổn định lâu dài, dự án áp dụng quy trình kiểm soát chất lượng và bảo mật tự động hiện đại:

- **Đánh giá mã nguồn tự động (PR Review)**: Sử dụng [CodeRabbit AI](https://coderabbit.ai) để phân tích chi tiết mọi thay đổi, giúp phát hiện sớm các lỗi logic và trường hợp biên trước khi phát hành.
- **Tối ưu hóa mã nguồn**: [Sourcery AI](https://sourcery.ai) liên tục rà soát mã nguồn để đề xuất các cấu trúc Python sạch, hiệu quả và chuẩn mực hơn.
- **Phân tích tĩnh & Bảo mật**: [CodeQL](https://codeql.github.com) thực hiện quét chuyên sâu để nhận diện các rủi ro bảo mật tiềm ẩn, đảm bảo mã nguồn tuân thủ các quy chuẩn an toàn.
- **Quy trình kiểm thử nghiêm ngặt**:
  - **[Ruff](https://github.com/astral-sh/ruff)**: Định dạng và kiểm tra lỗi (linting) Python tốc độ cao cho mã nguồn nhất quán.
  - **[Ty](https://github.com/astral-sh/ty)** & **[Pyright](https://github.com/Microsoft/pyright)**: Kiểm tra kiểu dữ liệu hai lớp để phát hiện lỗi trước khi chạy và duy trì sự ổn định của API.
  - **[Pytest](https://github.com/pytest-dev/pytest)**: Bộ kiểm thử tự động đảm bảo các tính năng luôn vận hành ổn định và không bị lỗi hồi quy (regression).
  - **[Interrogate](https://github.com/econchick/interrogate)**: Bắt buộc viết chú thích (docstring) đầy đủ cho tất cả các hàm và lớp để code luôn dễ đọc, dễ hiểu.
  - **[Prettier](https://github.com/prettier/prettier)**: Duy trì định dạng nhất quán cho các tệp tài liệu và cấu hình.

> [!NOTE]
> Mọi kết quả từ các công cụ tự động đều được quản trị viên dự án trực tiếp rà soát và xác nhận kỹ lưỡng, đảm bảo sự ổn định cao nhất cho người dùng.

---

## Đóng góp

Sự đóng góp từ cộng đồng là yếu tố cốt lõi giúp các dự án mã nguồn mở phát triển mạnh mẽ và sáng tạo hơn. Mọi đóng góp của bạn đều được **ghi nhận và trân trọng**.

> [!IMPORTANT]
> Môi trường phát triển cho dự án này là **Linux**. Nếu bạn sử dụng Windows, vui lòng cài đặt và sử dụng [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/vi-vn/windows/wsl/install), vì bộ công cụ phát triển và kiểm thử (test suite & development tools) được tối ưu hóa để chạy trên Linux.
>
> Các thư viện phụ thuộc và tiến trình thực thi của dự án được quản lý thông qua `uv`.

- **Nếu bạn tìm thấy lỗi**, hãy giúp dự án hoàn thiện hơn bằng cách [mở một issue](https://github.com/luuquangvu/assist-canonicalizer/issues).
- **Nếu bạn muốn đóng góp mã nguồn**, hãy fork kho lưu trữ và tạo Pull Request (đảm bảo mã nguồn của bạn đã vượt qua các [bước kiểm tra chất lượng](#chất-lượng-mã-nguồn--bảo-mật) phía trên).

---

## Bản quyền

Dự án được phát hành dưới **Giấy phép MIT**. Xem tệp [LICENSE](LICENSE) để biết thêm thông tin chi tiết.

## Hỗ trợ dự án

Nếu thấy dự án này hữu ích, sự ủng hộ của bạn là nguồn động lực lớn để mình tiếp tục hoàn thiện nó tốt hơn nữa. Cảm ơn bạn! ❤️

[![Donate via PayPal](https://img.shields.io/badge/Ủng%20hộ-PayPal-blue.svg?style=for-the-badge&logo=paypal)](https://www.paypal.me/luuquangvu89)
