## Cấu hình chốt lời, cắt lỗ cho tiểu khoản

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getPnlConfigsAccount"></span>

### `GET /accounts/{accountNo}/pnl-configs`

Lấy cấu hình chốt lời, cắt lỗ của tiểu khoản giao dịch, chỉ áp dụng với giao dịch phái sinh.

<h3 id="getpnlconfigsaccount-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|object|false|Loại thị trường |
|X-API-Key|header|object|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|object|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|object|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|Phiên bản API |
|accountNo|path|integer|true|Số tiểu khoản giao dịch|

#### Detailed descriptions

**marketType**: Loại thị trường 
- DERIVATIVE: Deal phái sinh
Hiện tại chỉ hỗ trợ phái sinh

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs \
  -H 'Accept: application/json' \
  -H 'X-API-Key: [object Object]' \
  -H 'X-Aux-Date: [object Object]' \
  -H 'X-Signature: [object Object]' \
  -H 'version: string'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: [object Object]
X-Aux-Date: [object Object]
X-Signature: [object Object]
version: string

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
        "X-API-Key": []string{"[object Object]"},
        "X-Aux-Date": []string{"[object Object]"},
        "X-Signature": []string{"[object Object]"},
        "version": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript

const headers = {
  'Accept':'application/json',
  'X-API-Key':{},
  'X-Aux-Date':{},
  'X-Signature':{},
  'version':'string'
};

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```python
import requests
headers = {
  'Accept': 'application/json',
  'X-API-Key': {},
  'X-Aux-Date': {},
  'X-Signature': {},
  'version': 'string'
}

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/pnl-configs");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

> Example responses

> 200 Response

```json
{
  "accountNo": "0001179019",
  "configs": {
    "takeProfit": {
      "enabled": true,
      "strategy": "PNL_RATE",
      "rate": 0.4,
      "deltaPrice": 7,
      "orderMethod": "FASTEST",
      "orderDeltaPrice": 2
    },
    "stopLoss": {
      "enabled": true,
      "strategy": "DELTA_PRICE",
      "rate": -0.6,
      "deltaPrice": 10.1,
      "orderMethod": "DELTA_PRICE",
      "orderDeltaPrice": 10.5,
      "trailingEnabled": true
    }
  },
  "createdDate": "2026-05-03T15:49:17.867357Z",
  "modifiedDate": "2026-08-28T08:34:50.6179Z"
}
```

<h3 id="getpnlconfigsaccount-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» accountNo|string|false|none|Số tiểu khoản chứng khoán|
|» configs|object|false|none|Cấu hình chốt lời và cắt lỗ của vị thế|
|»» takeProfit|object|false|none|Cấu hình chốt lời (Take Profit)|
|»»» enabled|boolean|false|none|Bật/tắt chức năng chốt lời|
|»»» strategy|string|false|none|Chiến lược kích hoạt chốt lời<br>- PNL_RATE: Kích hoạt theo tỷ lệ %<br>- DELTA_PRICE: Kích hoạt theo mức chênh lệch giá|
|»»» rate|number(double)|false|none|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = PNL_RATE. Giá trị phải lớn hơn 0|
|»»» deltaPrice|integer(int32)|false|none|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt chốt lời khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»»» orderMethod|string|false|none|Phương thức đặt lệnh khi điều kiện chốt lời được kích hoạt.<br>- FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua<br>- DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt|
|»»» orderDeltaPrice|integer(int32)|false|none|Biên độ giá đặt lệnh khi orderMethod = DELTA_PRICE. Bằng 0 khi orderMethod = FASTEST|
|»» stopLoss|object|false|none|Cấu hình cắt lỗ (Stop Loss)|
|»»» enabled|boolean|false|none|Bật/tắt chức năng cắt lỗ|
|»»» strategy|string|false|none|Chiến lược kích hoạt cắt lỗ<br>- PNL_RATE: Kích hoạt theo tỷ lệ %<br>- DELTA_PRICE: Kích hoạt theo mức chênh lệch giá|
|»»» rate|number(double)|false|none|Tỷ lệ % so với giá hòa vốn, dùng để kích hoạt cắt lỗ khi strategy = PNL_RATE. Giá trị hợp lệ trong khoảng [-1.0, 0)|
|»»» deltaPrice|number(double)|false|none|Mức chênh lệch giá so với giá hòa vốn, dùng để kích hoạt cắt lỗ khi strategy = DELTA_PRICE. Giá trị phải lớn hơn 0|
|»»» orderMethod|string(double)|false|none|Phương thức đặt lệnh khi điều kiện cắt lỗ được kích hoạt.<br>- FASTEST: Lệnh khớp ngay với giá đặt là giá trần/sàn tùy theo chiều vị thế Bán/Mua<br>- DELTA_PRICE: Lệnh đặt theo biên độ giá so với giá kích hoạt|
|»»» orderDeltaPrice|number(float)|false|none|Biên độ giá đặt lệnh khi orderMethod = DELTA_PRICE. Bằng 0 khi orderMethod = FASTEST|
|»»» trailingEnabled|boolean|false|none|Bật/tắt cơ chế Trailing Stop. Lưu ý cơ chế này chỉ hoạt động khi chức năng cắt lỗ được bật|
|» createdDate|string(date-time)|false|none|Thời điểm cấu hình được tạo|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật cấu hình gần nhất|

Status Code **400**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» code|string|false|none|none|
|» message|string|false|none|none|
|» status|integer|false|none|none|

Status Code **500**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» code|string|false|none|none|
|» message|string|false|none|none|
|» status|integer|false|none|none|
