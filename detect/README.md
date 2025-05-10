# Сэжигтэй үйлдэл илрүүлэх API

Энэ нь сэжигтэй үйлдлүүдийг AI загвараар илрүүлэх Flask API юм. API нь хүний сэжигтэй үйлдэл болон авто ослыг илрүүлэх боломжтой.

## Системийн шаардлага

- Python 3.8+
- pip
- Хангалттай GPU санах ой (CUDA дэмжигдсэн видео карттай)

## Суулгах

1. Энэ төслийг клонлох:

```
git clone <repository_url>
cd detect
```

2. Виртуал орчин үүсгэх (зөвлөмж):

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Шаардлагатай сангуудыг суулгах:

```
pip install -r requirements.txt
```

4. YOLOv8 загвар татах:

```
pip install -U ultralytics
```

## API ажиллуулах

API серверийг эхлүүлэхийн тулд дараах командыг ажиллуулна:

```
python api.py
```

Сервер 5000 портод эхлэнэ. Сервер ажиллаж эхэлмэгц терминал дээр дараах мессежүүд харагдана:

```
🚀 Flask API server starting...
🔄 Моделиудыг ачааллаж байна...
✅ YOLOv8 загвар ачаалагдлаа
✅ R(2+1)D загвар ачаалагдлаа
✅ Кинетик классууд ачаалагдлаа
✅ DETR загвар ачаалагдлаа
```

## API Endpoint-ууд

### Статус шалгах

**Хүсэлт:**

```
GET /status
```

**Хариулт:**

```json
{
  "status": "online",
  "version": "1.0.0",
  "models": {
    "yolo": true,
    "action": true,
    "accident": true
  },
  "timestamp": "2023-07-01T12:34:56.789Z"
}
```

### Видео танилт хийх

**Хүсэлт:**

```
POST /api/detect
Content-Type: multipart/form-data

video: [видео файл]
```

**Хариулт:**

```json
{
  "success": true,
  "video": "example.mp4",
  "frames_processed": 250,
  "alerts_count": 3,
  "results": [
    {
      "frame": 45,
      "type": "violent_action",
      "action": "punching person (boxing)",
      "confidence": 0.85,
      "bbox": [120, 80, 220, 280]
    },
    {
      "frame": 128,
      "type": "violent_action",
      "action": "wrestling",
      "confidence": 0.72,
      "bbox": [150, 100, 300, 350]
    },
    {
      "frame": 215,
      "type": "traffic_accident",
      "action": "accident",
      "confidence": 0.91,
      "bbox": [200, 150, 400, 300]
    }
  ]
}
```

## Сэрэмжлүүлэг хүлээн авах

API нь илрүүлсэн сэжигтэй үйлдэл бүрийг дараах URL руу илгээнэ:

```
http://localhost:3000/api/alerts
```

Системийн `API_URL` хувьсагчийг өөрчлөн хаяг солих боломжтой.

## NextJS аппликейштай холбогдох

NextJS аппликейшны `api/process-video/route.ts` файл нь энэхүү Flask API руу хүсэлт илгээж хариуг хүлээн авах боломжийг олгоно.

## Алдаа шийдвэрлэх

1. **Модель ачаалахад алдаа гарах:**

   - Шаардлагатай сангууд суусан эсэхийг шалгах
   - CUDA үндсэн/хувилбар зөрөөтэй эсэхийг шалгах

2. **Видео боловсруулахад алдаа гарах:**

   - Видеоны хэмжээ, өргөтгөл зөв эсэхийг шалгах
   - GPU санах ой хангалттай эсэхийг шалгах

3. **API серверт холбогдохгүй байх:**
   - 5000 порт нээлттэй эсэхийг шалгах
   - Файервол тохиргоо зөв эсэхийг шалгах
