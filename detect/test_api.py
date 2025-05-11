#!/usr/bin/env python3
"""
Видео боловсруулах API-г тестлэх энгийн скрипт.
"""

import os
import sys
import requests
import time
import json
from pathlib import Path

# API URL
API_URL = "http://localhost:5000"

def test_status():
    """API статус шалгах"""
    print("📡 API статус шалгаж байна...")
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            print("✅ API статус амжилттай:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ API статус алдаа: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API статус шалгах алдаа: {str(e)}")
        return False

def test_video_upload(video_path):
    """Видео файл илгээх тест"""
    if not os.path.exists(video_path):
        print(f"❌ Видео файл олдсонгүй: {video_path}")
        return False
    
    print(f"📤 Видео файлыг илгээж байна: {video_path}")
    
    try:
        # Файлыг нээж multipart/form-data хүсэлт үүсгэх
        with open(video_path, "rb") as f:
            start_time = time.time()
            
            files = {"video": (os.path.basename(video_path), f, "video/mp4")}
            data = {"location": "TEST_LOCATION"}
            
            print("🔄 API хүсэлт илгээж байна... (Түр хүлээнэ үү)")
            response = requests.post(f"{API_URL}/api/detect", files=files, data=data)
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ Видео боловсруулалт амжилттай! ({elapsed_time:.1f} секунд)")
                result = response.json()
                print(f"📊 Боловсруулсан кадр: {result.get('frames_processed', '?')}")
                print(f"⚠️ Илрүүлсэн сэрэмжлүүлэг: {result.get('alerts_count', 0)}")
                
                # Сэрэмжлүүлэг хэвлэх
                alerts = result.get("results", [])
                if alerts:
                    print("\n--- Сэрэмжлүүлэгүүд ---")
                    for i, alert in enumerate(alerts):
                        print(f"{i+1}. Төрөл: {alert.get('type')}, Үйлдэл: {alert.get('action')}, Итгэл: {alert.get('confidence', 0):.2f}")
                
                return True
            else:
                print(f"❌ API хүсэлт алдаатай: {response.status_code}")
                print(f"Хариу: {response.text}")
                return False
    except Exception as e:
        print(f"❌ API хүсэлт илгээх алдаа: {str(e)}")
        return False

def main():
    """Үндсэн функц"""
    print("=== Видео боловсруулах API тест ===\n")
    
    # API статус шалгах
    if not test_status():
        print("\n❌ API статус шалгахад алдаа гарлаа. API ажиллаж байгаа эсэхийг шалгана уу!")
        sys.exit(1)
    
    print("\n🔎 Видео файл хайж байна...")
    
    # Тестлэх видео файл хайх
    test_videos = [
        "./videoplayback.mp4",
        "./detect/videoplayback.mp4",
        "./uploads/test_video.mp4"
    ]
    
    video_path = None
    for path in test_videos:
        if os.path.exists(path):
            video_path = path
            break
    
    # Хэрэв тестлэх видео олдоогүй бол аргументаас шалгах
    if video_path is None and len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"❌ Өгөгдсөн видео файл олдсонгүй: {video_path}")
            video_path = None
    
    if video_path is None:
        print("❌ Тестлэх видео файл олдсонгүй. Файлын замыг аргументаар өгнө үү:")
        print(f"    python {sys.argv[0]} /замыг/видео/файлд.mp4")
        sys.exit(1)
    
    # Видео файл илгээх
    print("\n--- Видео боловсруулах тест ---")
    test_video_upload(video_path)

if __name__ == "__main__":
    main() 