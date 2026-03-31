import cv2
from roboflow import Roboflow
import time
import os
import datetime

# --- 1. ตั้งค่าโฟลเดอร์เก็บภาพ ---
SAVE_PATH = os.path.join(os.getcwd(), "QC_DEFECTS")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# --- 2. ฟังก์ชันสำหรับลบภาพที่เก่าเกิน 1 ชั่วโมง ---
def cleanup_old_images(folder_path, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # ตรวจสอบว่าเป็นไฟล์ (ไม่ใช่โฟลเดอร์)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f" [Auto-Cleanup] ลบไฟล์เก่า: {filename}")
                except Exception as e:
                    print(f" Error deleting file: {e}")

# --- 3. เชื่อมต่อ Roboflow API ---
rf = Roboflow(api_key="pxmjp3ToWaUbcY7r07sO")
project = rf.workspace().project("inspection-1fmfy")
model = project.version(1).model

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
freeze_until = 0 
last_cleanup_time = time.time() # เก็บเวลาที่ล้างข้อมูลล่าสุด

print(f"--- ระบบ QC เริ่มตรวจสอบ (ระบบจะลบภาพเก่าทุก 1 ชม.) ---")

while cap.isOpened():
    current_time = time.time()
    
    # --- ส่วนตรวจสอบและลบไฟล์ทุกๆ 5 นาที (เพื่อประหยัดทรัพยากร) ---
    if current_time - last_cleanup_time > 300: # เช็คทุก 5 นาที
        cleanup_old_images(SAVE_PATH, max_age_seconds=3600)
        last_cleanup_time = current_time

    if current_time > freeze_until:
        success, frame = cap.read()
        if not success: break
        
        results = model.predict(frame, confidence=25).json()

        if len(results["predictions"]) > 0:
            # เจอตำหนิ (FAIL 01)
            output_value = "01"
            print(f"RESULT: {output_value}")
            
            save_frame = frame.copy()
            for det in results["predictions"]:
                conf_percent = det["confidence"] * 100
                x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                
                cv2.rectangle(save_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 3)
                label = f"DENT {conf_percent:.1f}%"
                cv2.putText(save_frame, label, (int(x-w/2), int(y-h/2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # บันทึกภาพพร้อมชื่อไฟล์ตามเวลา
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            file_full_path = os.path.join(SAVE_PATH, f"defect_{timestamp}.jpg")
            cv2.imwrite(file_full_path, save_frame)
            
            display_frame = save_frame
            freeze_until = current_time + 20.0 
        else:
            # PASS 10
            output_value = "10"
            print(f"RESULT: {output_value}")
            display_frame = frame.copy()
            cv2.putText(display_frame, "STATUS: PASS (10)", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow("QC Inspection System", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()