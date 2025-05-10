import json
import cv2
import torch
import numpy as np
import requests
from collections import deque
from datetime import datetime
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO

# ----------------- [1] Config -----------------

# Зөрчилтэй үйлдлүүдийн whitelist
VIOLENT_CLASSES = {
    "punching person (boxing)",
    "wrestling",
    "slapping",
    "drop kicking",
    "side kick",
    "headbutting",
    "sword fighting"
}

# API endpoint
API_URL = "http://localhost:3000/api/alerts"

# ----------------- [2] Model ачаалах -----------------

# YOLO загвар
yolo_model = YOLO("yolov8n.pt")

# R(2+1)D загвар
action_model = r2plus1d_18(pretrained=True)
action_model.eval()

# Class.json файлыг ачаална
# with open("class.json", "r") as f:
with open("C:/Users/hp/Desktop/Dev Hackaton/detect/class.json", "r") as f:
    KINETICS_CLASSES = json.load(f)

# Трансформаци
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

# Frame buffer
frame_buffer = deque(maxlen=16)

# ----------------- Шинэ функц: Next.js API руу мэдээлэл илгээх -----------------
def send_alert_to_server(action, confidence):
    try:
        now = datetime.now()
        payload = {
            "action": action,
            "confidence": confidence,
            "timestamp": now.isoformat(),
            "location": "CAM_01"
        }
        
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Сэрэмжлүүлэг амжилттай илгээгдлээ: {action}")
        else:
            print(f"❌ Сэрэмжлүүлэг илгээхэд алдаа гарлаа: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Сэрэмжлүүлэг илгээхэд алдаа гарлаа: {str(e)}")

# Камер эхлүүлэх
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            cls_id = int(result.boxes.cls[i])
            label = result.names[cls_id]
            if label == 'person':
                x1, y1, x2, y2 = map(int, box)
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
                        label_text = f"{action} ({conf:.2f})"

                        # ----------------- [3] Зөрчил шалгах -----------------
                        if action in VIOLENT_CLASSES and conf > 0.10:
                            # Next.js API руу мэдээлэл илгээх
                            send_alert_to_server(action, conf)
                            
                            # Консол дээр мэдээлэл үзүүлэх
                            print(f"[ALERT] ⚠️ Dangerous action detected: {action} ({conf:.2f})")
                            print(f"Timestamp: {datetime.now()} | Location: CAM_01")

                        # ----------------- [4] UI харуулах -----------------
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 + R(2+1)D Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
