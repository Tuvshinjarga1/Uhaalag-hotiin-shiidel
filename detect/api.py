import os
import json
import time
import cv2
import torch
import numpy as np
import requests
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

# ----------------- [1] Config -----------------

VIOLENT_CLASSES = {
    "punching person (boxing)",
    "wrestling",
    "slapping", 
    "drop kicking",
    "side kick",
    "headbutting",
    "sword fighting"
}

API_URL = "http://localhost:3000/api/alerts"

# Flask аппликейшн тохируулга
app = Flask(__name__)
CORS(app)  # Бүх домайнаас хүсэлт зөвшөөрөх

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB хязгаарлалт

# Upload хавтас үүсгэх
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ----------------- [2] Model ачаалах -----------------
print("🚀 Flask API server starting...")
print("🔄 Моделиудыг ачааллаж байна...")

try:
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLOv8 загвар ачаалагдлаа")
except Exception as e:
    print(f"❌ YOLOv8 загвар ачаалахад алдаа: {str(e)}")
    yolo_model = None

try:
    action_model = r2plus1d_18(pretrained=True)
    action_model.eval()
    print("✅ R(2+1)D загвар ачаалагдлаа")
except Exception as e:
    print(f"❌ R(2+1)D загвар ачаалахад алдаа: {str(e)}")
    action_model = None

try:
    with open("C:/Users/hp/Desktop/Dev Hackaton/detect/class.json", "r") as f:
        KINETICS_CLASSES = json.load(f)
    print("✅ Кинетик классууд ачаалагдлаа")
except Exception as e:
    print(f"❌ Класс жагсаалт ачаалахад алдаа: {str(e)}")
    KINETICS_CLASSES = {}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# DETR загвар 
try:
    processor = DetrImageProcessor.from_pretrained("hilmantm/detr-traffic-accident-detection")
    accident_model = DetrForObjectDetection.from_pretrained("hilmantm/detr-traffic-accident-detection")
    accident_model.eval()
    print("✅ DETR загвар ачаалагдлаа")
except Exception as e:
    print(f"❌ DETR загвар ачаалахад алдаа: {str(e)}")
    processor = None
    accident_model = None

# ----------------- [3] Сэрэмжлүүлэг илгээх -----------------

def send_alert_to_server(action, confidence, location="CAM_01"):
    try:
        now = datetime.now()
        payload = {
            "action": action,
            "confidence": confidence,
            "timestamp": now.isoformat(),
            "location": location
        }
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"✅ Сэрэмжлүүлэг амжилттай илгээгдлээ: {action}")
            return True
        else:
            print(f"❌ Илгээхэд алдаа гарлаа: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Илгээхэд алдаа: {str(e)}")
        return False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, location_id="CAM_01"):
    """
    Видео файлыг боловсруулж, сэжигтэй үйлдлүүдийг илрүүлэх
    """
    if not os.path.exists(video_path):
        return {"error": f"Файл олдсонгүй: {video_path}"}
    
    if yolo_model is None or action_model is None or accident_model is None:
        return {"error": "Загваруудыг ачаалахад алдаа гарсан тул видео боловсруулах боломжгүй"}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Видеог нээхэд алдаа гарлаа: {video_path}"}
    
    frame_count = 0
    alerts = []
    frame_buffer = deque(maxlen=16)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 5 != 0:  # 5 кадр тутамд боловсруулах
                continue
                
            results = yolo_model.predict(frame)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    cls_id = int(result.boxes.cls[i])
                    label = result.names[cls_id]
                    confidence = result.boxes.conf[i].item()
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Хүний үйлдэл шалгах
                    if label == 'person' and x2-x1 > 60 and y2-y1 > 100:  # Хэт жижиг биетийг алгасах
                        roi = frame[y1:y2, x1:x2]
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            tensor = transform(roi_rgb)
                            frame_buffer.append(tensor)
                            
                            if len(frame_buffer) == 16:
                                clip = torch.stack(list(frame_buffer), dim=1).unsqueeze(0)
                                with torch.no_grad():
                                    logits = action_model(clip)
                                    probs = torch.softmax(logits, dim=1)
                                    pred_id = torch.argmax(probs, dim=1).item()
                                    conf = probs[0][pred_id].item()
                                
                                action = KINETICS_CLASSES.get(str(pred_id), "Unknown")
                                
                                if action in VIOLENT_CLASSES and conf > 0.10:
                                    alert = {
                                        "frame": frame_count,
                                        "type": "violent_action",
                                        "action": action,
                                        "confidence": float(conf),
                                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                    }
                                    alerts.append(alert)
                                    send_alert_to_server(action, conf, location_id)
                    
                    # Машины осол шалгах
                    elif label == "car" and x2-x1 > 80:  # Хэт жижиг машинуудыг алгасах
                        car_roi = frame[y1:y2, x1:x2]
                        
                        if car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
                            try:
                                pil_image = Image.fromarray(cv2.cvtColor(car_roi, cv2.COLOR_BGR2RGB))
                                inputs = processor(images=pil_image, return_tensors="pt")
                                
                                with torch.no_grad():
                                    outputs = accident_model(**inputs)
                                
                                target_size = torch.tensor([pil_image.size[::-1]])
                                results_detr = processor.post_process_object_detection(
                                    outputs, target_sizes=target_size, threshold=0.85)[0]
                                
                                for score, pred_label, bbox in zip(results_detr["scores"], 
                                                                 results_detr["labels"], 
                                                                 results_detr["boxes"]):
                                    label_text = accident_model.config.id2label[pred_label.item()]
                                    score_value = float(score)
                                    
                                    if label_text == "accident":
                                        alert = {
                                            "frame": frame_count,
                                            "type": "traffic_accident",
                                            "action": label_text,
                                            "confidence": score_value,
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                        }
                                        alerts.append(alert)
                                        send_alert_to_server(label_text, score_value, location_id)
                                        
                            except Exception as e:
                                print(f"❌ DETR осол шалгах алдаа: {str(e)}")
    
    except Exception as e:
        cap.release()
        return {"error": f"Видео боловсруулахад алдаа гарлаа: {str(e)}"}
    
    cap.release()
    
    return {
        "success": True,
        "video": os.path.basename(video_path),
        "frames_processed": frame_count,
        "alerts_count": len(alerts),
        "results": alerts
    }

# ----------------- [4] API endpoints -----------------

@app.route('/status', methods=['GET'])
def status():
    """Серверийн статус хариулах"""
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "models": {
            "yolo": yolo_model is not None,
            "action": action_model is not None,
            "accident": accident_model is not None
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_video():
    """
    Видео файл хүлээн авч илрүүлэлт хийх
    """
    # Видео файл байгаа эсэхийг шалгах
    if 'video' not in request.files:
        return jsonify({"error": "Видео файл илгээгдээгүй байна"}), 400
    
    file = request.files['video']
    
    # Хоосон файлын нэр шалгах
    if file.filename == '':
        return jsonify({"error": "Файлын нэр хоосон байна"}), 400
    
    # Зөвшөөрөгдсөн файлын төрөл шалгах
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Файлын төрөл дэмжигддэггүй. Дараах төрлүүд зөвшөөрөгдөнө: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Хэрэглэгч файлын нэрийг баталгаажуулах
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    
    try:
        file.save(file_path)
        print(f"✅ Видео файл хадгалагдлаа: {file_path}")
        
        # Видео боловсруулалт хийх
        location_id = request.form.get("location", "UPLOAD_01")
        result = process_video(file_path, location_id)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Файл хадгалахад алдаа гарлаа: {str(e)}"}), 500

if __name__ == '__main__':
    print(f"Flask сервер 5000 порт дээр эхэллээ...")
    app.run(debug=True, host='0.0.0.0', port=5000) 