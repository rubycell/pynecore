# Sandbox

Sandbox là môi trường thử nghiệm được DNSE cung cấp dành riêng cho người dùng và đối tác muốn kiểm thử tích hợp với hệ thống LightSpeed API (OpenAPI DNSE) trước khi triển khai trên môi trường thực.

---

## Sandbox là gì?

Sandbox của DNSE OpenAPI hoạt động tương tự môi trường thực Production nhưng hoàn toàn tách biệt:

- **Dữ liệu mô phỏng** — Thông tin tài khoản, số dư, lệnh giao dịch, danh mục đều là dữ liệu giả lập. 
- **Bộ Key riêng biệt** — Sandbox API Key và Sandbox API Secret chỉ hoạt động trên môi trường Sandbox, không dùng được ở môi trường thực và ngược lại.
- **Vòng đời lệnh tự động** — Lệnh đặt qua Sandbox tự động chuyển trạng thái theo vòng đời: `PendingNew` → `New` → `PartiallyFilled` → `Filled`, mô phỏng luồng khớp lệnh thực tế.
- **Giới hạn sử dụng** — Có rate limit nhất định, phù hợp cho kiểm thử tích hợp, không dùng để benchmark hiệu năng hay chạy chiến lược giao dịch.

> **Lưu ý:** Dữ liệu Sandbox không phản ánh thị trường thực. Không sử dụng kết quả từ Sandbox để ra quyết định đầu tư.

---

## Use cases

| Tình huống | Mô tả |
|---|---|
| **Kiểm thử tích hợp API** | Xác thực luồng kết nối, xác thực, và xử lý response |
| **Kiểm thử vòng đời lệnh** | Quan sát vòng đời lệnh tự động chuyển trạng thái mà không cần tài khoản tiền thật |
| **Phát triển và debug** | Thử nghiệm xử lý lỗi, retry logic, và các edge case một cách an toàn |
| **Kiểm thử WebSocket** | Kết nối WebSocket Sandbox để nhận event mô phỏng theo thời gian thực |
| **Onboarding đối tác mới** | Đối tác tích hợp có thể kiểm thử toàn bộ luồng mà không cần thao tác trên Production |

---

## Đăng ký Sandbox

**⚠️ Điều kiện:** Sandbox chỉ khả dụng cho tài khoản đã đăng ký thành công LightSpeed API (Production). Nếu chưa đăng ký, khách hàng vui lòng <a href="https://developers.dnse.com.vn/docs/guide/intro/register_guide">tham khảo hướng dẫn tại đây</a> và hoàn tất đăng ký OpenAPI trước khi thực hiện các bước dưới đây.

#### Bước 1 — Tại màn hình LightSpeed API DNSE

Truy cập mục LightSpeed API DNSE tại [trang web giao dịch trực tuyến](https://entradex.dnse.com.vn/thong-tin-ca-nhan/light-speed) chính thức của DNSE.

Cuộn xuống phần **"Bạn muốn thử nghiệm API trước khi triển khai?"**

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-cai-dat.png)](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-cai-dat.png)
</div>

#### Bước 2 — Thực hiện đăng ký

Khách hàng chọn **Đăng ký Sandbox** ở cuối trang.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-luu-y-su-dung.png)](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-luu-y-su-dung.png)
</div>

Đọc kỹ các lưu ý trong popup, tích vào ô xác nhận, sau đó chọn **Xác nhận đăng ký Sandbox**.

#### Bước 3 — Lưu bộ Key Sandbox 

Sau khi đăng ký thành công, hệ thống sẽ hiển thị thông tin quan trọng Sandbox API Key và Sandbox API Secret để kết nối.

<div className="guideImg">

[![Locale Dropdown](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-api-key.png)](https://cdn.dnse.com.vn/dnse-assets/EntradeX/G3-1/sandbox-api-key.png)
</div>

> ⚠️ **Sandbox Secret chỉ hiển thị đúng một lần** ngay sau khi đăng ký. Sao chép và lưu lại ngay trước khi rời trang.

---

## Kết nối Sandbox

Sandbox sử dụng cùng cấu trúc xác thực và cùng endpoint path như môi trường production. Người dùng chỉ cần thay đổi **base URL** và **key xác thực**.

| | Production | Sandbox |
|---|---|---|
| **REST API** | `https://openapi.dnse.com.vn` | `https://sb-openapi.dnse.com.vn` |
| **WebSocket** | `wss://ws-openapi.dnse.com.vn` | `wss://ws-sb-openapi.dnse.com.vn` |
| **API Key** | Production API Key | Sandbox API Key |
| **API Secret** | Production API Secret | Sandbox API Secret |

---

## Sử dụng REST API

Sandbox sử dụng cùng cơ chế xác thực như môi trường production: Mỗi request cần đính kèm `X-Api-Key`, `X-Signature`, và `Date` trong Headers.

**Điểm khác biệt:**

- **Base URL Sandbox: https://sb-openapi.dnse.com.vn**
- **Credentials: Sandbox API Keys, Sandbox API Secret**

Tham khảo <a href="https://developers.dnse.com.vn/docs/guide/intro/authentication">quy trình tạo Signature tại đây.</a>

#### Bước 1 — Lấy thông tin Sandbox Account 

Trước khi đặt lệnh, người dùng cần có số tiểu khoản, số dư tiền và gói vay thực tế của tài khoản Sandbox — các giá trị này khác hoàn toàn với môi trường production.

| Thông tin cần lấy | Endpoint |
|---|---|
| Danh sách tiểu khoản (`accountNo`) | `GET /accounts` |
| Thông tin tiền (`accountBalance`) | `GET /accounts/{accountNo}/balances` |
| Danh sách gói vay (`loanPackageId`) | `GET /accounts/{accountNo}/loan-packages` |
| Sức mua sức bán (`ppse`) | `GET /accounts/{accountNo}/ppse` |

Gọi lần lượt 4 endpoint trên để có đủ thông tin trước khi thực hiện đặt lệnh.

#### Bước 2 — Lấy Trading Token

Gọi <a href="https://developers.dnse.com.vn/docs/guide/intro/authentication">Endpoint xác thực OTP</a> để lấy Trading Token. Trên Sandbox, OTP được mô phỏng — không gửi mã thực về SmartOTP hay Email. Truyền `otpType` là `email_otp` hoặc `smart_otp` (cả hai đều được chấp nhận) và `passcode` là **666666**.

Hệ thống trả về Trading Token có hiệu lực 8 giờ:

```json lines
{ "trading-token": "7ceef658-9f01-414e-8b3e-faa77bb9061e" }
```

#### Bước 3 — Đặt lệnh

Đính kèm Trading Token vào Header và gọi Endpoint đặt lệnh. Hiện tại Sandbox chỉ hỗ trợ đặt lệnh thường `NORMAL`.

<details>
  <summary>Ví dụ Request Đặt lệnh Sandbox</summary>

```http
POST /accounts/{accountNo}/orders
Host: sb-openapi.dnse.com.vn
x-api-key: {sandbox_api_key}
x-Signature: Signature keyId="{sandbox_api_key}",algorithm="hmac-sha256",headers="(request-target) date",signature="{ENCODED_SIGNATURE}",nonce="{NONCE},
Date: {RFC1123_datetime}
trading-token: {sandbox_trading_token}
Content-Type: application/json
{
  "symbol": "41I1G9000",
  "side": "NB",
  "orderType": "LO",
  "quantity": 15,
  "price": 1300,
  "loanpackageId": 2026
}
```
</details>

#### Bước 4 — Theo dõi trạng thái lệnh

Sau khi đặt lệnh thành công, hệ thống Sandbox tự động mô phỏng vòng đời lệnh theo thứ tự:

```
PendingNew → New → PartiallyFilled → Filled
```

Theo dõi bằng cách polling Endpoint chi tiết lệnh, hoặc subscribe WebSocket để nhận event theo thời gian thực (xem phần WebSocket bên dưới).

---

## Sử dụng WebSocket

Kết nối WebSocket Sandbox để nhận dữ liệu mô phỏng giao dịch theo thời gian thực, bao gồm cập nhật trạng thái lệnh và vị thế mô phỏng.

**Base URL: wss://ws-sb-openapi.dnse.com.vn**

Tham khảo <a href="https://developers.dnse.com.vn/docs/guide/market-data/trading_connect">thông tin Trading WebSocket tại đây.</a>

Hiện tại Sandbox hỗ trợ kênh giữ liệu giao dịch:

- Order Event: `order.{market_type}.{encoding}`
- Position Event: `position.{market_type}.{encoding}`

> Dữ liệu WebSocket Sandbox là mô phỏng — không phản giao dịch hay tài sản thực tế.

---

## Giới hạn của Sandbox

- Không thể dùng để đánh giá hiệu quả chiến lược hoặc thuật toán giao dịch
- Dữ liệu Sandbox có thể được reset định kỳ
- Không có kết nối giữa Sandbox và môi trường Production