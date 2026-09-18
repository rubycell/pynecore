## Đảo vị thế

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="reversePosition"></span>

### `POST /positions/{positionId}/reverse`

Đóng toàn bộ vị thế phái sinh hiện tại và mở ngay vị thế mới theo hướng ngược lại. Lệnh LO ngược chiều, khối lượng đặt gấp đôi khối lượng mở của vị thế hiện tại, giá đặt bằng giá sàn (đảo Mua→Bán) hoặc giá trần (đảo Bán→Mua).
- Chỉ áp dụng cho giao dịch phái sinh
    
- Hỗ trợ các vị thế có khối lượng mở tối đa 250 HĐ.
    
- Không áp dụng cho các vị thế trong trường hợp: đang bị bán xử lý (force sell), đang chờ đóng hoặc đang chờ đảo vị thế.

<h3 id="reverseposition-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|object|false|Loại thị trường |
|X-API-Key|header|object|false|API Key được cấp khi đăng ký dịch vụ|
|X-Signature|header|object|false|Chữ ký xác thực yêu cầu|
|X-Aux-Date|header|object|false|Thời gian thực hiện yêu cầu|
|trading-token|header|object|false|Token đặt lệnh|
|version|header|string|false|API version (YYYY-MM-DD)|
|positionId|path|object|true|Id vị thế|

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Danh sách Deal cơ sở
- DERIVATIVE: Danh sách Deal phái sinh
Hiện tại chỉ hỗ trợ phái sinh

> Code samples

```shell
# You can also use wget
curl -X POST https://openapi.dnse.com.vn/positions/{positionId}/reverse \
  -H 'Accept: application/json' \
  -H 'X-API-Key: [object Object]' \
  -H 'X-Signature: [object Object]' \
  -H 'X-Aux-Date: [object Object]' \
  -H 'trading-token: [object Object]' \
  -H 'version: string'

```

```http
POST https://openapi.dnse.com.vn/positions/{positionId}/reverse HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: [object Object]
X-Signature: [object Object]
X-Aux-Date: [object Object]
trading-token: [object Object]
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
        "X-Signature": []string{"[object Object]"},
        "X-Aux-Date": []string{"[object Object]"},
        "trading-token": []string{"[object Object]"},
        "version": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "https://openapi.dnse.com.vn/positions/{positionId}/reverse", data)
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
  'X-Signature':{},
  'X-Aux-Date':{},
  'trading-token':{},
  'version':'string'
};

fetch('https://openapi.dnse.com.vn/positions/{positionId}/reverse',
{
  method: 'POST',

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
  'X-Signature': {},
  'X-Aux-Date': {},
  'trading-token': {},
  'version': 'string'
}

r = requests.post('https://openapi.dnse.com.vn/positions/{positionId}/reverse', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/positions/{positionId}/reverse");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
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
  "id": 446,
  "side": "NS",
  "accountNo": "0001179019",
  "symbol": "41I1G9000",
  "price": 1804.2,
  "quantity": 46,
  "orderType": "LO",
  "loanPackageId": 2279,
  "createdDate": "2026-09-14T03:07:16.806",
  "modifiedDate": "2026-09-14T03:07:16.806"
}
```

<h3 id="reverseposition-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» id|integer(int32)|false|none|Id lệnh giao dịch|
|» side|string|false|none|Chiều đặt lệnh<br>- NB: Mua<br>- NS: Bán|
|» accountNo|string|false|none|Số tiểu khoản|
|» symbol|string|false|none|Mã chứng khoán|
|» price|number(double)|false|none|Giá đặt|
|» quantity|integer(int32)|false|none|Khối lượng đặt|
|» orderType|string|false|none|Loại lệnh<br>- LO: Lệnh giới hạn|
|» loanPackageId|integer(int32)|false|none|Mã gói vay|
|» createdDate|string(date-time)|false|none|Thời điểm tạo|
|» modifiedDate|string(date-time)|false|none|Thời điểm cập nhật|

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

<h1 id="openapi-v2-spec-260730-market-data">market-data</h1>
