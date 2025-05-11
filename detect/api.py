import os
import json
import time
import cv2
import torch
import numpy as np
import requests
import base64
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

# Сэжигтэй үйлдлийн дахин мэдэгдэх хугацааны хязгаарлалт (секундээр)
ALERT_COOLDOWN = 10  # 10 секундын "хөргөлт"

# Осол/хүчирхийллийн үргэлжлэх минимум хугацаа (секунд)
MIN_INCIDENT_DURATION = 3  # Хамгийн багадаа 3 секунд үргэлжилсэн үйлдлийг л мэдэгдэх

# Машины хөдөлгөөний хянагч
CAR_MOTION_THRESHOLD = 0.01  # Хөдөлгөөн дэмжих хязгаар (пиксел өөрчлөлтийн хувь) - уулзварт удаан хөдөлгөөнд зориулж бууруулсан
MIN_MOTION_HISTORY = 5  # Машины хөдөлгөөний түүхийг хадгалах хугацаа (кадраар)
ACCIDENT_DETECTION_THRESHOLD = 0.83  # Ослыг тодорхойлох доод хязгаар - бууруулсан
ACCIDENT_MIN_FRAMES = 2  # Осол хэмээн тодорхойлохын тулд хамгийн багадаа шаардагдах кадрын тоо
STOPPED_CAR_MAX_MOTION = 0.005  # Зогссон машин гэж үзэх хамгийн их хөдөлгөөн

# Сүүлийн мэдэгдэл хадгалах
last_alerts = {}  # {action_type: last_timestamp}

# Анх илрүүлсэн үйлдлүүдийн хугацааг хадгалах
first_detection_times = {}  # {action_key: first_time}

# Нотлох баримт зураг хадгалах хавтас
EVIDENCE_DIR = './evidence'
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

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

def encode_image_to_base64(image):
    """
    Зургийг base64 форматруу хөрвүүлэх
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_alert_to_server(action, confidence, location="CAM_01", evidence_images=None):
    try:
        now = datetime.now()
        
        # Хөргөлтийн шалгалт хийх - ижил үйлдэл ALERT_COOLDOWN секундын дотор давтагдаж байвал алгасах
        action_key = f"{action}_{location}"
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
            "location": location
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
                img_filename = f"{EVIDENCE_DIR}/{action}_{location}_{timestamp}_{i}.jpg"
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Машины хөдөлгөөнийг хянах функц
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
    
    # Машины байршлын түүхийг хадгалах
    if location_history is not None and len(location_history) > 0:
        # Сүүлийн байршилтай харьцуулж байршил өөрчлөгдсөн эсэхийг шалгах
        prev_positions = location_history[-min(len(location_history), MIN_MOTION_HISTORY):]
        position_changes = []
        
        for prev_pos in prev_positions:
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
    
    # Тухайн видео файлын хүрээнд мэдэгдсэн үйлдлүүдийн сүүлийн цаг
    local_alert_times = {}
    
    # Зургийн нотолгоотой холбоотой хувьсагчууд
    capture_evidence = {}  # {action_key: [next_capture_frame, [existing_frames]]}
    
    # Үйлдлүүдийн эхний илрүүлэлтийн хугацаа
    local_first_detections = {}  # {action_key: first_time}
    
    # Өмнөх кадрыг хадгалах (машины хөдөлгөөнийг хянахад шаардлагатай)
    prev_frame = None
    
    # Машины байршлийн түүхийг хадгалах
    car_position_history = {}  # {car_id: [positions]}
    # Ослын шалгалтын тоологч хадгалах
    car_accident_count = {}  # {car_id: accident_detection_count}
    # Машины өмнөх ROI хадгалах
    car_previous_rois = {}  # {car_id: [previous_rois]}
    car_tracking_count = 0  # Машиныг түр зуур дугаарлах
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = time.time()
            
            # Нотлох зураг авах шаардлагатай эсэхийг шалгах
            for action_key, (next_frame, frames) in list(capture_evidence.items()):
                if frame_count == next_frame:
                    # Хоёрдахь зургийг нэмэх
                    frames.append(frame.copy())
                    if len(frames) == 2:
                        # Аль нэг үйлдэлд 2 зураг цугларсан бол мэдэгдэл явуулах
                        action, location = action_key.split('_', 1)
                        alert_info = next((a for a in alerts if a["action"] == action), None)
                        
                        # Илрүүлэлтийн нийт үргэлжилсэн хугацааг тооцоолох
                        first_detected_time = local_first_detections.get(action_key, 0)
                        detection_duration = current_time - first_detected_time
                        
                        # Хэрэв хангалттай удаан (9+ секунд) үргэлжилсэн бол мэдэгдэл явуулах
                        if detection_duration >= MIN_INCIDENT_DURATION and alert_info:
                            print(f"✅ {action} үйлдэл {detection_duration:.1f} секунд үргэлжилсэн, мэдэгдэл явуулж байна")
                            
                            # Нотлох дүрсэн дээр үргэлжилсэн хугацааг нэмж харуулах
                            cv2.putText(frames[1], f"{detection_duration:.1f} секунд үргэлжилсэн", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            send_alert_to_server(
                                action, 
                                alert_info["confidence"], 
                                location, 
                                frames
                            )
                        else:
                            print(f"⏱️ {action} үйлдэл {detection_duration:.1f} секунд үргэлжилсэн - хангалттай удаан биш ({MIN_INCIDENT_DURATION} сек шаардлагатай)")
                            
                        # Хураангуй жагсаалтаас цэвэрлэх
                        del capture_evidence[action_key]
                    else:
                        # Дараагийн зургийг 5 секундын дараа авах
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        next_frame = frame_count + int(fps * 5)  # 5 секундын дараа
                        capture_evidence[action_key] = (next_frame, frames)
            
            # 5 кадр тутамд боловсруулах
            if frame_count % 5 != 0:
                continue
                
            results = yolo_model.predict(frame)
            
            # Энэ кадр дахь машинуудыг хадгалах
            current_cars = []
            
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
                                    # Хөргөлтийн шалгалт хийх - дотоод
                                    now = time.time()
                                    action_key = f"{action}_{location_id}"
                                    can_alert = True
                                    
                                    if action_key in local_alert_times:
                                        time_diff = now - local_alert_times[action_key]
                                        if time_diff < ALERT_COOLDOWN:
                                            can_alert = False
                                            print(f"⏱️ {action} үйлдэл видео доторх {time_diff:.1f} секундын дотор илрүүлсэн тул алгаслаа")
                                    
                                    if can_alert:
                                        local_alert_times[action_key] = now
                                        
                                        # Эхний илрүүлэлтийн хугацааг хадгалах
                                        if action_key not in local_first_detections:
                                            local_first_detections[action_key] = now
                                            print(f"⏱️ {action} үйлдэл эхлэлийн хугацааг бүртгэлээ: {now}")
                                        
                                        alert = {
                                            "frame": frame_count,
                                            "type": "violent_action",
                                            "action": action,
                                            "confidence": float(conf),
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                                        }
                                        alerts.append(alert)
                                        
                                        # Нотлох зургийг авах - эхний зураг
                                        evidence_frame = frame.copy()
                                        # Боксыг зурах
                                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        label_text = f"{action} ({conf:.2f})"
                                        cv2.putText(evidence_frame, label_text, (x1, y1-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        
                                        # Дараагийн зураг авах хугацааг тооцоолох
                                        fps = cap.get(cv2.CAP_PROP_FPS)
                                        next_frame = frame_count + int(fps * 5)  # 5 секундын дараа
                                        capture_evidence[action_key] = (next_frame, [evidence_frame])
                    
                    # Машины осол шалгах
                    elif label == "car" and x2-x1 > 80:  # Хэт жижиг машинуудыг алгасах
                        car_roi = frame[y1:y2, x1:x2]
                        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Машинд түр ID олгох (хялбар tracking)
                        car_id = None
                        min_distance = 50  # Машиныг ялгах зай хэмжээ
                        
                        # Мөн машин мөн эсэхийг шалгах
                        for c_id, positions in car_position_history.items():
                            if len(positions) > 0:
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
                        car_previous_rois[car_id].append(car_roi.copy())
                        # Хэмжээг хязгаарлах
                        if len(car_previous_rois[car_id]) > 5:  # Сүүлийн 5 кадр хадгалах
                            car_previous_rois[car_id] = car_previous_rois[car_id][-5:]
                        
                        if car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
                            try:
                                # Машин хөдөлгөөнтэй эсэхийг шалгах - байршлийн түүхийг ашиглан
                                is_moving, motion_score = check_vehicle_motion(
                                    prev_frame, frame, [x1, y1, x2, y2], car_position_history[car_id])
                                
                                # Хөдөлгөөнгүй машин бол осол шалгахгүй алгасах
                                if not is_moving:
                                    print(f"🚙 Зогсож буй машин илэрлээ (хөдөлгөөний оноо: {motion_score:.3f}), осол шалгахгүй")
                                    continue
                                
                                print(f"🚗 Хөдөлгөөнтэй машин илэрлээ (хөдөлгөөний оноо: {motion_score:.3f}), осол шалгаж байна...")
                                
                                pil_image = Image.fromarray(cv2.cvtColor(car_roi, cv2.COLOR_BGR2RGB))
                                inputs = processor(images=pil_image, return_tensors="pt")
                                
                                with torch.no_grad():
                                    outputs = accident_model(**inputs)
                                
                                target_size = torch.tensor([pil_image.size[::-1]])
                                results_detr = processor.post_process_object_detection(
                                    outputs, target_sizes=target_size, threshold=0.80)[0]  # Доод хязгаарыг бууруулж, илүү олон тохиолдол шалгах
                                
                                for score, pred_label, bbox in zip(results_detr["scores"], 
                                                                 results_detr["labels"], 
                                                                 results_detr["boxes"]):
                                    label_text = accident_model.config.id2label[pred_label.item()]
                                    score_value = float(score)
                                    
                                    if label_text == "accident":
                                        # Ослыг нэмэлтээр шалгах (хуурамч эерэг үр дүнг бууруулахын тулд)
                                        is_accident, reason = verify_accident(
                                            car_roi, 
                                            motion_score, 
                                            car_previous_rois.get(car_id, []),
                                            car_position_history.get(car_id, [])
                                        )
                                        
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
                                        action_key = f"{label_text}_{location_id}"
                                        can_alert = True
                                        
                                        if action_key in local_alert_times:
                                            time_diff = now - local_alert_times[action_key]
                                            if time_diff < ALERT_COOLDOWN:
                                                can_alert = False
                                                print(f"⏱️ {label_text} үйлдэл видео доторх {time_diff:.1f} секундын дотор илрүүлсэн тул алгаслаа")
                                        
                                        if can_alert:
                                            local_alert_times[action_key] = now
                                            
                                            # Эхний илрүүлэлтийн хугацааг хадгалах
                                            if action_key not in local_first_detections:
                                                local_first_detections[action_key] = now
                                                print(f"⏱️ {label_text} үйлдэл эхлэлийн хугацааг бүртгэлээ: {now}")
                                            
                                            alert = {
                                                "frame": frame_count,
                                                "type": "traffic_accident",
                                                "action": label_text,
                                                "confidence": score_value,
                                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                                "motion_score": motion_score,
                                                "accident_detection_count": car_accident_count[car_id]
                                            }
                                            alerts.append(alert)
                                            
                                            # Нотлох зургийг авах - эхний зураг
                                            evidence_frame = frame.copy()
                                            # Боксыг зурах
                                            cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                            info_text = f"{label_text} ({score_value:.2f}, motion: {motion_score:.2f})"
                                            cv2.putText(evidence_frame, info_text, (x1, y1-10), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                            
                                            # Дараагийн зураг авах хугацааг тооцоолох
                                            fps = cap.get(cv2.CAP_PROP_FPS)
                                            next_frame = frame_count + int(fps * 5)  # 5 секундын дараа
                                            capture_evidence[action_key] = (next_frame, [evidence_frame])
                                            
                                            print(f"[ALERT 🚨] Машины осол илэрлээ: {label_text} ({score_value:.2f}) - {car_accident_count[car_id]} кадрт баталгаажсан")
                                        
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
            
            # Одоогийн кадрыг дараагийн удаад ашиглахаар хадгалах
            prev_frame = frame.copy()
    
    except Exception as e:
        cap.release()
        return {"error": f"Видео боловсруулахад алдаа гарлаа: {str(e)}"}
    
    cap.release()
    
    # Шаардлагатай нотлох зураг авч чадаагүй үйлдлүүдийг хүлээж байсан бол API рүү мэдэгдэл явуулахгүйгээр дуусгах
    for action_key, (_, frames) in capture_evidence.items():
        print(f"⚠️ {action_key} үйлдэлд бүрэн нотлох зураг авч чадаагүй")
    
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