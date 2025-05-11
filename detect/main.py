import json
import cv2
import torch
import numpy as np
import requests
import base64
import os
from collections import deque
from datetime import datetime
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time

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

# Сэжигтэй үйлдлийн дахин мэдэгдэх хугацааны хязгаарлалт (секундээр)
ALERT_COOLDOWN = 10  # 10 секундын "хөргөлт"

# Сүүлийн мэдэгдэл хадгалах
last_alerts = {}  # {action_type: last_timestamp}

# Нотлох баримт зураг хадгалах хавтас
EVIDENCE_DIR = './evidence'
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

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

def encode_image_to_base64(image):
    """
    Зургийг base64 форматруу хөрвүүлэх
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_alert_to_server(action, confidence, evidence_images=None):
    try:
        now = datetime.now()
        
        # Хөргөлтийн шалгалт хийх - ижил үйлдэл ALERT_COOLDOWN секундын дотор давтагдаж байвал алгасах
        action_key = f"{action}_CAM_01"
        if action_key in last_alerts:
            last_time = last_alerts[action_key]
            time_diff = (now - last_time).total_seconds()
            if time_diff < ALERT_COOLDOWN:
                print(f"⏱️ {action} үйлдэл {time_diff:.1f} секундын өмнө илгээгдсэн тул алгаслаа")
                return False
        
        # Үйлдлийн цаг хугацааг хадгалах
        last_alerts[action_key] = now
        
        payload = {
            "action": action,
            "confidence": confidence,
            "timestamp": now.isoformat(),
            "location": "CAM_01"
        }
        
        # Нотлох зураг байвал нэмэх
        if evidence_images and len(evidence_images) > 0:
            encoded_images = []
            for i, img in enumerate(evidence_images):
                # Зургийг base64 болгох
                encoded_img = encode_image_to_base64(img)
                encoded_images.append(encoded_img)
                
                # Зургийг файл болгон хадгалах
                timestamp = int(time.time())
                img_filename = f"{EVIDENCE_DIR}/{action}_CAM_01_{timestamp}_{i}.jpg"
                cv2.imwrite(img_filename, img)
                print(f"✅ Нотлох зураг хадгалагдлаа: {img_filename}")
            
            # Нотлох зургийг payload-д нэмэх
            payload["evidence_images"] = encoded_images
        
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


# ----------------- [4] Камер унших -----------------
# ----------------- [4] Камер унших -----------------
video_path = "./videoplayback.mp4"  # ← Энд өөрийн видео файлын нэрийг оруулна
# video_path = "./V_1.mp4"  # ← Энд өөрийн видео файлын нэрийг оруулна
cap = cv2.VideoCapture(video_path)

# Фрейм тоолуур
frame_count = 0
# Видео-н локал хөргөлтийн механизм (фрейм хоорондын)
local_action_times = {}

# Зургийн нотолгоо цуглуулагч
evidence_collector = {}  # {action: [time_to_capture, [frames]]}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Нотлох зураг авах хугацаа болсон эсэхийг шалгах
    current_time = time.time()
    for action, (capture_time, frames) in list(evidence_collector.items()):
        if current_time >= capture_time:
            # Хоёр дахь зургийг нэмэх
            frames.append(frame.copy())
            if len(frames) == 2:
                # Дүрсний нотолгоо бүрдсэн, дохиолол явуулах
                print(f"📸 {action}-д нотлох 2 дахь зураг цуглуулсан")
                # Нөхцөл байдлыг тодорхойлсон текст нэмэх
                cv2.putText(frames[1], f"{action} - 5 сек үргэлжилсэн", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Найдвартай байдлын түвшинг авах
                confidence = 0.0
                # Тухайн action-тай холбоотой confidence хайх
                for key, time_value in local_action_times.items():
                    if key == action:
                        confidence = 0.85  # Хэрэв олдохгүй бол дундаж итгэх байдал оноох
                
                # Мэдэгдэл явуулах
                send_alert_to_server(action, confidence, frames)
                # Жагсаалтаас хасах
                del evidence_collector[action]
            else:
                # Дараагийн зураг авах хугацааг тохируулах (5 секунд)
                evidence_collector[action] = (current_time + 5.0, frames)
    
    # 5 кадр тутамд боловсруулах (ачааллыг багасгах)
    if frame_count % 5 != 0:
        continue
 
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
                            # Фрейм хоорондын локал хөргөлтийн шалгалт
                            now = time.time()
                            action_key = f"{action}"
                            should_alert = True
                            
                            if action_key in local_action_times:
                                time_diff = now - local_action_times[action_key]
                                if time_diff < ALERT_COOLDOWN:
                                    should_alert = False
                                    print(f"⏱️ {action} үйлдэл {time_diff:.1f} секундын өмнө илэрсэн тул дахин мэдэгдэхгүй")
                            
                            if should_alert:
                                local_action_times[action_key] = now
                                
                                # Эхний нотлох зургийг авах
                                evidence_frame = frame.copy()
                                cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(evidence_frame, label_text, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.putText(evidence_frame, "Эхлэл", (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                # Нотлох зургийн цуглуулагчид нэмэх
                                if action not in evidence_collector:
                                    evidence_collector[action] = (now + 2.0, [evidence_frame])
                                
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
                                # Фрейм хоорондын локал хөргөлтийн шалгалт
                                now = time.time()
                                action_key = f"{label_text}"
                                should_alert = True
                                
                                if action_key in local_action_times:
                                    time_diff = now - local_action_times[action_key]
                                    if time_diff < ALERT_COOLDOWN:
                                        should_alert = False
                                        print(f"⏱️ {label_text} {time_diff:.1f} секундын өмнө илэрсэн тул дахин мэдэгдэхгүй")
                                
                                if should_alert:
                                    local_action_times[action_key] = now
                                    
                                    # Эхний нотлох зургийг авах
                                    evidence_frame = frame.copy()
                                    cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    acc_text = f"{label_text} ({score:.2f})"
                                    cv2.putText(evidence_frame, acc_text, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(evidence_frame, "Эхлэл", (10, 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Нотлох зургийн цуглуулагчид нэмэх
                                    if label_text not in evidence_collector:
                                        evidence_collector[label_text] = (now + 2.0, [evidence_frame])
                                    
                                    print(f"[ALERT 🚨] Машины осол илэрлээ: {label_text} ({score:.2f})")
 
                    except Exception as e:
                        print(f"❌ DETR осол шалгах алдаа: {str(e)}")
 
    cv2.imshow("YOLOv8 + R(2+1)D + DETR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()