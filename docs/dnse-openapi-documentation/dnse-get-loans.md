## Danh sách khoản vay

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getLoans"></span>

### `GET /accounts/{accountNo}/loans`

Lấy danh sách các khoản vay trên tiểu khoản theo bộ lọc. Hiện tại chỉ hỗ trợ các khoản vay cơ sở.

<h3 id="getloans-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|marketType|query|object|false|Loại thị trường |
|disbursementFrom|query|object|false|Ngày giải ngân bắt đầu (YYYY-MM-DD)|
|disbursementTo|query|object|false|Ngày giải ngân kết thúc (YYYY-MM-DD)|
|dueDateFrom|query|object|false|Ngày đến hạn bắt đầu (YYYY-MM-DD)|
|dueDateTo|query|object|false|Ngày đến hạn kết thúc (YYYY-MM-DD)|
|positionId|query|integer|false|Id vị thế|
|pageSize|query|object|false|Số bản ghi trên mỗi trang|
|pageIndex|query|object|false|Vị trí của trang hiện tại|
|X-API-Key|header|string|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|string|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|string|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
|accountNo|path|integer|true|Số tiểu khoản |

#### Detailed descriptions

**marketType**: Loại thị trường 
- STOCK: Danh sách vị thế cơ sở
- DERIVATIVE: Danh sách vị thế phái sinh

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/accounts/{accountNo}/loans \
  -H 'Accept: application/json' \
  -H 'X-API-Key: string' \
  -H 'X-Aux-Date: string' \
  -H 'X-Signature: string' \
  -H 'version: string'

```

```http
GET https://openapi.dnse.com.vn/accounts/{accountNo}/loans HTTP/1.1
Host: openapi.dnse.com.vn
Accept: application/json
X-API-Key: string
X-Aux-Date: string
X-Signature: string
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
        "X-API-Key": []string{"string"},
        "X-Aux-Date": []string{"string"},
        "X-Signature": []string{"string"},
        "version": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/accounts/{accountNo}/loans", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

```javascript

const headers = {
  'Accept':'application/json',
  'X-API-Key':'string',
  'X-Aux-Date':'string',
  'X-Signature':'string',
  'version':'string'
};

fetch('https://openapi.dnse.com.vn/accounts/{accountNo}/loans',
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
  'X-API-Key': 'string',
  'X-Aux-Date': 'string',
  'X-Signature': 'string',
  'version': 'string'
}

r = requests.get('https://openapi.dnse.com.vn/accounts/{accountNo}/loans', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/accounts/{accountNo}/loans");
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
  "loans": [
    {
      "id": 8117,
      "accountNo": "0001179019",
      "positionId": 8426,
      "symbol": "ACB",
      "disbursementDate": "2026-09-10",
      "dueDate": "2026-12-09",
      "initialRate": 1,
      "interestRate": 0.115,
      "overdueInterestRate": 0.18,
      "preferentialPeriod": 0,
      "preferentialInterestRate": 0,
      "principal": 1191245,
      "remainPrincipal": 1191245,
      "remainInterest": 1501
    }
  ],
  "pagination": {
    "pageIndex": 0,
    "pageSize": 20,
    "totalRecords": 2,
    "totalPages": 1
  }
}
```

<h3 id="getloans-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» loans|[object]|false|none|Danh sách các khoản vay|
|»» id|integer(int32)|false|none|ID khoản vay|
|»» accountNo|string|false|none|Số tiểu khoản|
|»» positionId|integer(int32)|false|none|ID vị thế liên kết với khoản vay|
|»» symbol|string|false|none|Mã chứng khoán|
|»» disbursementDate|string|false|none|Ngày giải ngân khoản vay|
|»» dueDate|string|false|none|Ngày đến hạn của khoản vay|
|»» initialRate|integer(int32)|false|none|Tỷ lệ cho vay ban đầu|
|»» interestRate|number(double)|false|none|Lãi suất áp dụng cho khoản vay|
|»» overdueInterestRate|number(double)|false|none|Lãi suất áp dụng khi khoản vay quá hạn|
|»» preferentialPeriod|integer(int32)|false|none|Thời gian áp dụng lãi suất ưu đãi|
|»» preferentialInterestRate|integer(int32)|false|none|Lãi suất ưu đãi áp dụng cho khoản vay|
|»» principal|integer(int32)|false|none|Giá trị dư nợ gốc ban đầu của khoản vay|
|»» remainPrincipal|integer(int32)|false|none|Giá trị dư nợ gốc còn lại|
|»» remainInterest|integer(int32)|false|none|Giá trị lãi vay còn lại|
|» pagination|object|false|none|Thông tin phân trang của danh sách kết quả.|
|»» pageIndex|integer(int32)|false|none|Chỉ số trang hiện tại, bắt đầu từ 0|
|»» pageSize|integer(int32)|false|none|Số lượng bản ghi tối đa trên mỗi trang|
|»» totalRecords|integer(int32)|false|none|Tổng số bản ghi|
|»» totalPages|integer(int32)|false|none|Tổng số trang|

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
