# Tài Liệu API Chatbot Y Tế

## Tổng Quan API

Base URL: `https://your-domain.com/api/v1`

### Xác Thực

Tất cả các requests cần include header:

```
Authorization: Bearer <token>
```

## Endpoints

### 1. Chat API

#### POST /chat

Gửi tin nhắn và nhận phản hồi từ chatbot.

**Request:**

```json
{
  "message": "Tôi bị đau đầu",
  "patient_id": "P123456"
}
```

**Response:**

```json
{
  "response": "Xin cho biết thêm về cơn đau đầu của bạn",
  "patient_id": "P123456",
  "state": "symptoms",
  "collected_info": {
    "symptoms": ["đau đầu"],
    "predictions": {
      "specialties": [
        {
          "specialty": "Thần kinh",
          "confidence": 0.85
        }
      ]
    }
  }
}
```

### 2. Trạng Thái Hàng Đợi

#### POST /queue-status

Kiểm tra trạng thái hàng đợi của một chuyên khoa.

**Request:**

```json
{
  "specialty": "Thần kinh"
}
```

**Response:**

```json
{
  "specialty": "Thần kinh",
  "current_number": 45,
  "waiting_time": 30
}
```

### 3. WebSocket

#### WS /ws

Kết nối WebSocket cho real-time chat.

**Connect:**

```javascript
const ws = new WebSocket("wss://your-domain.com/ws");
```

**Message Format:**

```json
{
  "message": "Tin nhắn từ người dùng",
  "patient_id": "P123456"
}
```

### 4. Bệnh Án

#### GET /medical-record/{patient_id}

Lấy bệnh án của bệnh nhân.

**Response:**

```json
{
  "patient_id": "P123456",
  "personal_info": {
    "name": "Nguyễn Văn A",
    "age": 30,
    "gender": "Nam"
  },
  "medical_info": {
    "symptoms": ["đau đầu", "sốt"],
    "diagnosis": "Đau đầu cấp tính",
    "recommendations": {
      "specialty": "Thần kinh",
      "treatments": ["Paracetamol"]
    }
  }
}
```

## Mã Lỗi

| Mã  | Mô tả                   |
| --- | ----------------------- |
| 200 | Thành công              |
| 400 | Lỗi dữ liệu đầu vào     |
| 401 | Chưa xác thực           |
| 403 | Không có quyền truy cập |
| 404 | Không tìm thấy          |
| 500 | Lỗi server              |

## Rate Limiting

- 100 requests/phút cho mỗi IP
- 1000 requests/giờ cho mỗi token

## Môi Trường

### Production

```
https://api.your-domain.com
```

### Staging

```
https://staging-api.your-domain.com
```

### Development

```
http://localhost:8000
```

## Các Chú Ý

1. Sử dụng HTTPS cho tất cả requests
2. Xử lý lỗi mạng và retry
3. Implement proper timeout
4. Validate dữ liệu đầu vào
