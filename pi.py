import cv2
import numpy as np
from flask import Flask, render_template, Response
from pypylon import pylon

# 1. เริ่มต้นระบบ Flask
app = Flask(__name__)

# 2. ตั้งค่า (Configuration)
CONFIG = {
    "camera_ip": "192.168.1.11",
    "exposure": 40040.0,
    "roi": {"x": 420, "y": 50, "w": 350, "h": 400},
    "sensitivity": 25,  # ค่า SENSITIVITY ที่คุณตั้งไว้
    "min_area": 5       # ค่า MIN_AREA ที่คุณตั้งไว้
}

# 3. จัดการกล้อง Basler
class BaslerCamera:
    def __init__(self):
        self.camera = None
        try:
            tl = pylon.TlFactory.GetInstance()
            devices = tl.EnumerateDevices()
            for d in devices:
                if d.GetIpAddress() == CONFIG["camera_ip"]:
                    self.camera = pylon.InstantCamera(tl.CreateDevice(d))
                    self.camera.Open()
                    self.camera.ExposureTimeRaw.SetValue(int(CONFIG["exposure"]))
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    print(f" เชื่อมต่อกล้อง Basler สำเร็จ! (IP: {CONFIG['camera_ip']})")
                    break
            if not self.camera:
                print(" ไม่พบกล้องที่ IP นี้ กรุณาตรวจสอบการเชื่อมต่อ")
        except Exception as e:
            print(f" Camera Error: {e}")

    def get_frame(self):
        if self.camera and self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                img = grab.Array
                grab.Release()
                return img
        return None

cam = BaslerCamera()

# 4. อัลกอริทึมตรวจจับรอย (DoG Filter)
def process_frame(frame):
    r = CONFIG["roi"]
    # ตัดเฉพาะพื้นที่แผ่นเตารีด
    roi_img = frame[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
    
    # ดึงความคมชัดของรอยออกมาด้วย Difference of Gaussians
    g1 = cv2.GaussianBlur(roi_img, (3, 3), 0)
    g2 = cv2.GaussianBlur(roi_img, (15, 15), 0)
    dog = cv2.subtract(g1, g2)
    
    _, thresh = cv2.threshold(dog, CONFIG["sensitivity"], 255, cv2.THRESH_BINARY)
    
    # วาดการแสดงผล
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(display, (r["x"], r["y"]), (r["x"]+r["w"], r["y"]+r["h"]), (0, 255, 0), 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > CONFIG["min_area"]:
            (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
            # วงกลมสีแดงรอบจุดที่เจอตำหนิ
            cv2.circle(display, (int(x_c)+r["x"], int(y_c)+r["y"]), int(radius)+5, (0, 0, 255), 2)
            
    return display

# 5. ส่วนควบคุมเว็บไซต์ (Web Interface)
@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = cam.get_frame()
        if frame is not None:
            processed = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 6. เริ่มการทำงาน
if __name__ == '__main__':
    print("--- ระบบ QC เริ่มทำงาน (กด Ctrl+C เพื่อหยุด) ---")
    print(" เปิด Browser ไปที่ http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)