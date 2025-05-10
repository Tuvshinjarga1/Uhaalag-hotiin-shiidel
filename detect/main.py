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
 
# ----------------- [2] Model ачаалах -----------------
 
yolo_model = YOLO("yolov8n.pt")
 
action_model = r2plus1d_18(pretrained=True)
action_model.eval()
 
with open("./class.json", "r") as f:
    KINETICS_CLASSES = json.load(f)
 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])
 
frame_buffer = deque(maxlen=16)
 
# DETR загвар (1 удаа ачаалах)
processor = DetrImageProcessor.from_pretrained("hilmantm/detr-traffic-accident-detection")
accident_model = DetrForObjectDetection.from_pretrained("hilmantm/detr-traffic-accident-detection")
accident_model.eval()
 
 
# ----------------- [3] Сэрэмжлүүлэг илгээх -----------------
 
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
            print(f"❌ Илгээхэд алдаа гарлаа: {response.status_code}")
    except Exception as e:
        print(f"❌ Илгээхэд алдаа: {str(e)}")
 
 
# ----------------- [4] Камер унших -----------------
# ----------------- [4] Камер унших -----------------
video_path = "./videoplayback.mp4"  # ← Энд өөрийн видео файлын нэрийг оруулна
cap = cv2.VideoCapture(video_path)
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
 
            x1, y1, x2, y2 = map(int, box)
 
            # ----------------- [5] Хүний үйлдэл шалгах -----------------
            if label == 'person':
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
 
                        if action in VIOLENT_CLASSES and conf > 0.10:
                            send_alert_to_server(action, conf)
                            print(f"[ALERT] ⚠️ Dangerous action detected: {action} ({conf:.2f})")
 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
 
            # ----------------- [6] Машины осол шалгах -----------------
            elif label == "car":
                print("🚗 Машин илэрлээ, осол шалгаж байна...")
                car_roi = frame[y1:y2, x1:x2]
 
                if car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(car_roi, cv2.COLOR_BGR2RGB))
                        inputs = processor(images=pil_image, return_tensors="pt")
                        with torch.no_grad():
                            outputs = accident_model(**inputs)
 
                        target_size = torch.tensor([pil_image.size[::-1]])
                        results_detr = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0.85)[0]
 
                        for score, pred_label, bbox in zip(results_detr["scores"], results_detr["labels"], results_detr["boxes"]):
                            label_text = accident_model.config.id2label[pred_label.item()]
                            if label_text == "accident":
                                print(f"[ALERT 🚨] Машины осол илэрлээ: {label_text} ({score:.2f})")
                                send_alert_to_server(label_text, float(score))
 
                    except Exception as e:
                        print(f"❌ DETR осол шалгах алдаа: {str(e)}")
 
    cv2.imshow("YOLOv8 + R(2+1)D + DETR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()