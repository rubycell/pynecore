## Lịch sử chỉ số

### Base URLs:
- **https://openapi.dnse.com.vn**

<span id="getMarketIndex"></span>

### `GET /price/{indexName}/market-index`

Truy vấn thông tin khối lượng khớp theo bước giá

<h3 id="getmarketindex-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|from|query|object|false|Thời gian bắt đầu (timestamp)|
|to|query|object|false|Thời gian kết thúc (timestamp) (không vượt quá 1 ngày)|
|limit|query|integer|false|none|
|nextPageToken|query|string|false|none|
|X-API-Key|header|object|false|API Key được cấp khi đăng ký dịch vụ|
|X-Aux-Date|header|object|false|Thời gian thực hiện yêu cầu|
|X-Signature|header|object|false|Chữ ký xác thực yêu cầu|
|version|header|string|false|API version (YYYY-MM-DD)|
|indexName|path|object|true|Chỉ sổ thị trường|

#### Detailed descriptions

**indexName**: Chỉ sổ thị trường
- HNX: Chỉ số sàn HNX 
- HNX30: Chỉ số Top 30 cổ phiếu sàn HNX 
- VN30: Chỉ số Top 30 cổ phiếu sàn HOSE 
- VNINDEX: Chỉ số sàn HOSE 
- UPCOM: Chỉ số sàn UPCOM
- VN100: Chỉ số Top 100 cổ phiếu sàn HOSE  
- VNXALLSHARE: Chỉ số các cổ phiếu chọn lọc sàn HOSE 
- VNDIVIDEND: Chỉ số nhóm cổ phiếu có tỷ suất cổ tức tăng trưởng 
- VN50GROWTH: Chỉ số nhóm 50 cổ phiếu tăng trưởng sàn HOSE 
- VNMITECH: Chỉ số nhóm cổ phiếu công nghệ

> Code samples

```shell
# You can also use wget
curl -X GET https://openapi.dnse.com.vn/price/{indexName}/market-index \
  -H 'Accept: application/json' \
  -H 'X-API-Key: [object Object]' \
  -H 'X-Aux-Date: [object Object]' \
  -H 'X-Signature: [object Object]' \
  -H 'version: string'

```

```http
GET https://openapi.dnse.com.vn/price/{indexName}/market-index HTTP/1.1
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
    req, err := http.NewRequest("GET", "https://openapi.dnse.com.vn/price/{indexName}/market-index", data)
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

fetch('https://openapi.dnse.com.vn/price/{indexName}/market-index',
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

r = requests.get('https://openapi.dnse.com.vn/price/{indexName}/market-index', headers = headers)

print(r.json())

```

```java
URL obj = new URL("https://openapi.dnse.com.vn/price/{indexName}/market-index");
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
  "marketIndices": [
    {
      "marketId": "STO",
      "indexName": "VNINDEX",
      "tradingSessionId": "99",
      "marketIndexClass": "HSX",
      "indexTypeCode": "001",
      "currencyCode": "VND",
      "transactionTime": "2026-08-19 15:05:00.108",
      "valueIndexes": 2735.41,
      "totalVolumeTraded": 2059678,
      "grossTradeAmount": 163.757653341,
      "contauctAccTrdVol": 257579,
      "contauctAccTrdVal": 17.11369539,
      "blkTrdAccTrdVol": 1802099,
      "blkTrdAccTrdVal": 146.643957951,
      "fluctuationUpperLimitIssueCount": 11,
      "fluctuationUpIssueCount": 19,
      "fluctuationSteadinessIssueCount": 11,
      "fluctuationDownIssueCount": 7,
      "fluctuationLowerLimitIssueCount": 9,
      "fluctuationUpIssueVolume": 1240687,
      "fluctuationSteadinessIssueVolume": 52400,
      "fluctuationDownIssueVolume": 184367,
      "highestValueIndexes": 2735.41,
      "lowestValueIndexes": 2681.97,
      "changedValue": 1003.3899999999999,
      "changedRatio": 57.93,
      "priorValueIndexes": 1732.02
    }
  ],
  "nextPageToken": "eyJ0cmFuc2FjdGlvblRpbWUiOiIyMDI2LTA4LTE5VDA4OjA1OjAwLjEwOFoiLCJpbmRleFR5cGVDb2RlIjoiMDAxIiwibWFya2V0SWQiOjZ9"
}
```

<h3 id="getmarketindex-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» marketIndices|[object]|false|none|none|
|»» marketId|string|false|none|Mã thị trường|
|»» indexName|string|false|none|Tên chỉ số thị trường|
|»» tradingSessionId|string|false|none|Mã phiên giao dịch hiện tại|
|»» marketIndexClass|string|false|none|Phân loại chỉ số|
|»» indexTypeCode|string|false|none|Mã loại chỉ số|
|»» currencyCode|string|false|none|Đơn vị tiền tệ|
|»» transactionTime|string|false|none|Thời gian giao dịch|
|»» valueIndexes|number(double)|false|none|Giá trị hiện tại của chỉ số|
|»» totalVolumeTraded|integer(int32)|false|none|Tổng khối lượng giao dịch trong ngày|
|»» grossTradeAmount|number(double)|false|none|Tổng giá trị giao dịch trong ngày|
|»» contauctAccTrdVol|integer(int32)|false|none|Tổng khối lượng giao dịch theo phương thức khớp lệnh|
|»» contauctAccTrdVal|number(double)|false|none|Tổng giá trị giao dịch theo phương thức khớp lệnh|
|»» blkTrdAccTrdVol|integer(int32)|false|none|Tổng khối lượng giao dịch theo phương thức thỏa thuận|
|»» blkTrdAccTrdVal|number(double)|false|none|Tổng giá trị giao dịch theo phương thức thỏa thuận|
|»» fluctuationUpperLimitIssueCount|integer(int32)|false|none|Số lượng mã tăng trần|
|»» fluctuationUpIssueCount|integer(int32)|false|none|Số lượng mã có giá tăng|
|»» fluctuationSteadinessIssueCount|integer(int32)|false|none|Số lượng mã có giá không đổi|
|»» fluctuationDownIssueCount|integer(int32)|false|none|Số lượng mã có giá giảm|
|»» fluctuationLowerLimitIssueCount|integer(int32)|false|none|Số lượng mã giảm sàn|
|»» fluctuationUpIssueVolume|integer(int32)|false|none|Tổng khối lượng giao dịch các mã có giá tăng|
|»» fluctuationSteadinessIssueVolume|integer(int32)|false|none|Tổng khối lượng giao dịch các mã có giá không đổi|
|»» fluctuationDownIssueVolume|integer(int32)|false|none|Tổng khối lượng giao dịch các mã có giá giảm|
|»» highestValueIndexes|number(double)|false|none|Giá cao nhất trong phiên|
|»» lowestValueIndexes|number(double)|false|none|Giá thấp nhất trong phiên|
|»» changedValue|number(double)|false|none|Giá trị thay đổi so với tham chiếu|
|»» changedRatio|number(double)|false|none|Tỷ lệ thay đổi (%)|
|»» priorValueIndexes|number(double)|false|none|Giá trị tham chiếu|
|» nextPageToken|string|false|none|Token dùng để phân trang kết quả|

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
