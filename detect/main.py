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

# Машины хөдөлгөөний хязгаар
CAR_MOTION_THRESHOLD = 0.01  # Хөдөлгөөн шалгах хязгаар - уулзварт удаан хөдөлгөөнд зориулж бууруулсан
MIN_MOTION_HISTORY = 5  # Машины хөдөлгөөний түүхийг хадгалах хугацаа (кадраар)
ACCIDENT_DETECTION_THRESHOLD = 0.83  # Ослыг тодорхойлох доод хязгаар - бууруулсан
ACCIDENT_MIN_FRAMES = 2  # Осол хэмээн тодорхойлохын тулд хамгийн багадаа шаардагдах кадрын тоо
STOPPED_CAR_MAX_MOTION = 0.005  # Зогссон машин гэж үзэх хамгийн их хөдөлгөөн

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

# Машины хөдөлгөөнийг шалгах функц
def check_vehicle_motion(prev_frame, current_frame, bbox, location_history=None):
    """
    Машин хөдөлгөөнтэй эсэхийг шалгах, удаан хөдөлгөөн болон уулзварын онцлогийг тооцох
    bbox: [x1, y1, x2, y2] - Машины координат
    location_history: Машины байршлийн түүх - уулзвар дээр байгаа эсэхийг тодорхойлоход ашиглана
    return: (хөдөлгөөнтэй эсэх, хөдөлгөөний оноо)
    """
    if prev_frame is None:
        return True, 1.0  # Өмнөх кадр байхгүй бол хөдөлгөөнтэй гэж үзэх
        
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Хязгаарыг шалгах
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(prev_frame.shape[0], y2)
    x2 = min(prev_frame.shape[1], x2)
    
    # ROI хэмжээ хэт жижиг эсвэл хязгаараас хальж байвал алгасах
    if y2 <= y1 or x2 <= x1:
        return True, 1.0
    
    # Өмнөх болон одоогийн кадраас тухайн хэсгийг авах
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = current_frame[y1:y2, x1:x2]
    
    # Харьцуулахын тулд саарал өнгөнд хөрвүүлэх
    if len(prev_roi.shape) > 2:
        prev_roi_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    else:
        prev_roi_gray = prev_roi
        
    if len(curr_roi.shape) > 2:
        curr_roi_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    else:
        curr_roi_gray = curr_roi
    
    # Кадр хоорондын зөрүүг тооцоолох
    frame_diff = cv2.absdiff(prev_roi_gray, curr_roi_gray)
    
    # Зөрүүг нөхцөл хангасан эсэхийг шалгахын тулд босготой харьцуулах
    _, thresh_diff = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)  # Илүү бага босго ашиглах (15)
    
    # Хөдөлгөөний оноог тооцоолох (өөрчлөгдсөн пикселийн хувь)
    motion_score = np.count_nonzero(thresh_diff) / (thresh_diff.shape[0] * thresh_diff.shape[1])
    
    # Хөдөлгөөнтэй гэж үзэх эсэх
    is_moving = motion_score > CAR_MOTION_THRESHOLD
    
    # Машины байршлын түүх бүртгэгдсэн бол хөдөлгөөний чиглэлийг тооцох
    if location_history is not None and len(location_history) > 0:
        # Сүүлийн байршилтай харьцуулж байршил өөрчлөгдсөн эсэхийг шалгах
        prev_positions = location_history[-min(len(location_history), MIN_MOTION_HISTORY):]
        position_changes = []
        
        for prev_pos in prev_positions:
            if prev_pos is None:  # Алдагдсан байршил алгасах
                continue
                
            prev_x, prev_y = prev_pos
            dx = center_x - prev_x
            dy = center_y - prev_y
            
            # Байршил хоорондын зөрүү 
            position_change = np.sqrt(dx*dx + dy*dy)
            position_changes.append(position_change)
        
        # Дундаж хөдөлгөөн
        if position_changes:
            avg_position_change = sum(position_changes) / len(position_changes)
            # Хэрэв машин бүр мөсөн хөдөлж байвал (удаан ч гэсэн), хөдөлгөөнтэй гэж тооцно
            if avg_position_change > 0.5:  # Хамгийн бага шаардагдах хөдөлгөөн (пиксел)
                is_moving = True
                
    # Уулзвар дээр удаан явж буй машиныг нэмж шалгах (хөдөлгөөн бага байх тул)
    if motion_score > 0.001 and motion_score <= CAR_MOTION_THRESHOLD:
        # Маш удаан хөдөлгөөнтэй машиныг ч хөдөлгөөнтэй гэж үзнэ
        print(f"🚖 Уулзвар дээрх удаан хөдөлгөөнтэй машин: {motion_score:.4f}")
        # Хөдөлж буй машин гэж үзэх 
        is_moving = True
    
    return is_moving, motion_score

# Ослын шинж чанарыг тодорхойлох нэмэлт функц
def verify_accident(car_roi, motion_score, previous_frames=None, position_history=None):
    """
    Машины ослыг бүрэн шалгах нэмэлт функц
    Энэ нь илүү нарийвчилсан шалгалт хийж, жирийн хөдөлгөөнтэй машиныг ослоос ялгана
    """
    # Тогтмол хөдөлгөөнтэй машин (хэт өндөр хөдөлгөөн) бол осол биш байх магадлал өндөр
    if motion_score > 0.5:
        return False, "Хэт өндөр хөдөлгөөнтэй (0.5+)"
    
    # Зогссон машин байж болзошгүй эсэхийг шалгах
    is_truly_stopped = motion_score < STOPPED_CAR_MAX_MOTION
    
    # Хэрэв машин бүрэн зогссон бол, энэ нь осол байж болох
    if is_truly_stopped and position_history is not None:
        # Зогсохын өмнөх хөдөлгөөнийг шалгах
        has_previous_motion = False
        
        # Хангалттай урт түүхтэй эсэхийг шалгах
        if len(position_history) >= 5:
            # Сүүлийн 5 байршлын өөрчлөлтийг шалгах
            positions = [p for p in position_history[-10:] if p is not None]
            if len(positions) >= 5:
                # Сүүлийн 5 байршлын өөрчлөлтийг тооцоолох
                position_changes = []
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    change = np.sqrt(dx*dx + dy*dy)
                    position_changes.append(change)
                
                # Нэг хэсэг нь хөдөлгөөнтэй байсан эсэх (гэнэт зогссон)
                has_previous_motion = any(change > 3.0 for change in position_changes)
                
                # Хэрэв өмнө нь хөдөлж байгаад гэнэт зогссон бол осол байх магадлал өндөр
                if has_previous_motion:
                    return True, "Гэнэт зогссон - өмнө нь хөдөлгөөнтэй байсан"
                    
        # Удаан хугацаанд зогссон машин (осол биш) эсэхийг шалгах
        # (хэрэв бид тодорхойлж чадахгүй бол эргэлзээтэй үед true буцаана)
    
    # Машины өнгөний хязгаарыг шалгах - улаан өнгө ихтэй байвал дохио, гэрэл байж болзошгүй
    if car_roi is not None:
        # BGR руу хөрвүүлэх
        hsv = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
        
        # Машины roi дахь улаан өнгийн хувийг тооцоолох
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_percentage = np.count_nonzero(red_mask) / (car_roi.shape[0] * car_roi.shape[1])
        
        # Хэрэв улаан өнгө 30%-с их байвал (дохио гэрэл гэх мэт)
        if red_percentage > 0.3:
            return False, f"Улаан өнгө ихтэй ({red_percentage:.2f})"
    
    # Хэрэв машин бүрэн зогссон бол энэ осол байж болно
    if is_truly_stopped:
        return True, "Бүрэн зогссон машин (осол байж болзошгүй)"
    
    # Хөдөлгөөн маш бага байвал (0.005-0.03) шалгах
    if motion_score < 0.03:
        return True, f"Маш бага хөдөлгөөнтэй ({motion_score:.4f}), осол байж болзошгүй"
    
    # Шалгалт амжилттай болсон (осол байж болзошгүй)
    return True, "Осол байж болзошгүй"

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
video_path = "./as.mp4"  # ← Энд өөрийн видео файлын нэрийг оруулна
# video_path = "./V_1.mp4"  # ← Энд өөрийн видео файлын нэрийг оруулна
cap = cv2.VideoCapture(video_path)

# Фрейм тоолуур
frame_count = 0
# Видео-н локал хөргөлтийн механизм (фрейм хоорондын)
local_action_times = {}

# Зургийн нотолгоо цуглуулагч
evidence_collector = {}  # {action: [time_to_capture, [frames]]}

# Өмнөх кадрыг хадгалах (хөдөлгөөн шалгахад ашиглана)
prev_frame = None

# Машины байршлийн түүхийг хадгалах
car_position_history = {}  # {car_id: [positions]}
car_tracking_count = 0  # Машиныг түр зуур дугаарлах
# Ослын шалгалтын тоологч хадгалах
car_accident_count = {}  # {car_id: accident_detection_count}
# Машины өмнөх ROI хадгалах
car_previous_rois = {}  # {car_id: [previous_rois]}

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
                # Нөхцөл байдлын түвшинг авах
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

    # Одоогийн кадр дахь машинуудыг хадгалах
    current_cars = []
 
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
                                    evidence_collector[action] = (now + 5.0, [evidence_frame])
                                
                                print(f"[ALERT] ⚠️ Dangerous action detected: {action} ({conf:.2f})")
 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
 
            # ----------------- [6] Машины осол шалгах -----------------
            elif label == "car":
                car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                # Машинд түр ID олгох (хялбар tracking)
                car_id = None
                min_distance = 50  # Машиныг ялгах зай хэмжээ
                
                # Мөн машин мөн эсэхийг шалгах
                for c_id, positions in car_position_history.items():
                    if len(positions) > 0 and positions[-1] is not None:
                        last_pos = positions[-1]
                        dist = np.sqrt((car_center[0] - last_pos[0])**2 + (car_center[1] - last_pos[1])**2)
                        if dist < min_distance:
                            car_id = c_id
                            break
                
                # Шинэ машин бол шинэ ID оноох
                if car_id is None:
                    car_id = f"car_{car_tracking_count}"
                    car_tracking_count += 1
                    car_position_history[car_id] = []
                
                # Байршлийн түүхийг шинэчлэх
                car_position_history[car_id].append(car_center)
                
                # Түүхийн хэмжээг хязгаарлах
                if len(car_position_history[car_id]) > MIN_MOTION_HISTORY * 3:
                    car_position_history[car_id] = car_position_history[car_id][-MIN_MOTION_HISTORY * 3:]
                
                # Машины өмнөх ROI хадгалах
                if car_id not in car_previous_rois:
                    car_previous_rois[car_id] = []
                
                # Түүхийг шинэчлэх
                car_previous_rois[car_id].append(frame[y1:y2, x1:x2].copy())
                # Хэмжээг хязгаарлах
                if len(car_previous_rois[car_id]) > 5:  # Сүүлийн 5 кадр хадгалах
                    car_previous_rois[car_id] = car_previous_rois[car_id][-5:]
                
                # Одоогийн кадр дахь машиныг хадгалах
                current_cars.append(car_id)
                
                # Машин хөдөлгөөнтэй эсэхийг шалгах - байршлийн түүхийг ашиглан
                is_moving, motion_score = check_vehicle_motion(
                    prev_frame, frame, [x1, y1, x2, y2], car_position_history[car_id])
                
                # Хөдөлгөөнгүй машин бол алгасах
                if not is_moving:
                    print(f"🚙 Зогсож буй машин илэрлээ (хөдөлгөөний оноо: {motion_score:.3f}), осол шалгахгүй")
                    continue
                    
                print(f"🚗 Хөдөлгөөнтэй машин илэрлээ (хөдөлгөөний оноо: {motion_score:.3f}), осол шалгаж байна...")
                car_roi = frame[y1:y2, x1:x2]
 
                if car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(car_roi, cv2.COLOR_BGR2RGB))
                        inputs = processor(images=pil_image, return_tensors="pt")
                        with torch.no_grad():
                            outputs = accident_model(**inputs)
 
                        target_size = torch.tensor([pil_image.size[::-1]])
                        results_detr = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0.80)[0]
 
                        for score, pred_label, bbox in zip(results_detr["scores"], results_detr["labels"], results_detr["boxes"]):
                            label_text = accident_model.config.id2label[pred_label.item()]
                            if label_text == "accident":
                                # Ослыг нэмэлтээр шалгах (хуурамч эерэг үр дүнг бууруулахын тулд)
                                is_accident, reason = verify_accident(car_roi, motion_score, car_previous_rois.get(car_id, []), car_position_history.get(car_id, []))
                                
                                if not is_accident:
                                    print(f"⚠️ Осол шалгалтаар буруу гарлаа: {reason}")
                                    
                                    # Машинд ослын тоологч байгаа эсэхийг шалгах
                                    if car_id not in car_accident_count:
                                        car_accident_count[car_id] = 0
                                        
                                    continue  # Нэмэлт шалгалтаар буруу гарсан тул алгасах
                                else:
                                    print(f"✅ Осол байж болзошгүй: {reason}")
                                
                                # Машинд ослын тоологч байгаа эсэхийг шалгах
                                if car_id not in car_accident_count:
                                    car_accident_count[car_id] = 0
                                    
                                # Осол илрүүлсэн тоог нэмэгдүүлэх
                                car_accident_count[car_id] += 1
                                
                                # Хэрэв хангалттай тооны кадрт осол илрээгүй бол мэдэгдэл үүсгэхгүй
                                if car_accident_count[car_id] < ACCIDENT_MIN_FRAMES:
                                    print(f"⏱️ Осол илрүүлэлт {car_accident_count[car_id]}/{ACCIDENT_MIN_FRAMES} (Баталгаажуулж байна...)")
                                    continue
                                    
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
                                    acc_text = f"{label_text} ({score:.2f}, motion: {motion_score:.2f})"
                                    cv2.putText(evidence_frame, acc_text, (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(evidence_frame, f"Эхлэл: {reason}", (10, 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Нотлох зургийн цуглуулагчид нэмэх
                                    if label_text not in evidence_collector:
                                        evidence_collector[label_text] = (now + 5.0, [evidence_frame])
                                    
                                    print(f"[ALERT 🚨] Машины осол илэрлээ: {label_text} ({score:.2f}) - {car_accident_count[car_id]} кадрт баталгаажсан, шалтгаан: {reason}")
 
                    except Exception as e:
                        print(f"❌ Машины хөдөлгөөн эсвэл DETR осол шалгах алдаа: {str(e)}")
    
    # Хадгалсан машины түүхийг цэвэрлэх - одоогийн кадрт байхгүй машиныг устгах
    for car_id in list(car_position_history.keys()):
        if car_id not in current_cars:
            # Осол тоологчийг цэвэрлэх
            if car_id in car_accident_count:
                del car_accident_count[car_id]
            
            # ROI түүхийг цэвэрлэх
            if car_id in car_previous_rois:
                del car_previous_rois[car_id]
            
            # Машин кадраас гарч, хэсэг хугацаанд буцаж ирээгүй бол устгах
            car_position_history[car_id].append(None)  # None-г нэмж тэмдэглэх
            
            # Хэрэв машин удаан хугацаанд алга болсон бол устгах
            none_count = sum(1 for pos in car_position_history[car_id][-5:] if pos is None)
            if none_count >= 5:  # 5 удаа дараалан алга болсон бол устгах
                del car_position_history[car_id]
    
    # Одоогийн кадрыг хадгалах
    prev_frame = frame.copy()
 
    cv2.imshow("YOLOv8 + R(2+1)D + DETR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()