import cv2
import numpy as np
from pypylon import pylon
from flask import Flask, Response
import threading
import time

# --- [CONFIG] ---
SENSITIVITY = 15          # ลดค่าลงนิดหน่อยเพื่อให้มองข้ามเงาจางๆ ของขอบ
MIN_AREA, MAX_AREA = 25, 350
CAMERA_IP = "192.168.1.11"

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def camera_thread():
    global output_frame
    try:
        factory = pylon.TlFactory.GetInstance()
        info = pylon.CDeviceInfo()
        info.SetIpAddress(CAMERA_IP)
        camera = pylon.InstantCamera(factory.CreateDevice(info))
        camera.Open()

        # --- [PYLON SETTINGS] ---
        camera.ExposureTimeRaw.SetValue(18000) # ลด Exposure ลงเล็กน้อยเพื่อไม่ให้ขอบฟุ้ง
        camera.GainRaw.SetValue(0)
        try:
            camera.BinningHorizontal.SetValue(2)
            camera.BinningVertical.SetValue(2)
        except: pass

        camera.GevSCPSPacketSize.SetValue(1500)
        camera.GevSCPD.SetValue(8000)
        camera.AcquisitionFrameRateAbs.SetValue(20.0)

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grabResult and grabResult.GrabSucceeded():
                raw_img = converter.Convert(grabResult).GetArray()
                frame = cv2.resize(raw_img, (480, 360))

                # --- [ROI Management] ---
                h, w = frame.shape[:2]
                # บีบ ROI เข้ามาอีกเล็กน้อย (25%) เพื่อหลบขอบเตารีดที่อาจจะแลบเข้ามา
                x1, y1, x2, y2 = int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)
                roi = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # ใช้ Gaussian Blur เพื่อลบความคมของขอบ
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)

                _, thresh = cv2.threshold(blurred, SENSITIVITY, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                defect_count = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if MIN_AREA < area < MAX_AREA:
                        # --- [กรองขอบเตารีดด้วย Aspect Ratio] ---
                        x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                        aspect_ratio = float(w_r) / h_r

                        # ตำหนิจริงๆ สัดส่วนกว้าง/สูง ควรจะใกล้เคียงกัน (ไม่เป็นเส้นยาว)
                        # ถ้ากว้างกว่าสูง 3 เท่า หรือ สูงกว่ากว้าง 3 เท่า -> สันนิษฐานว่าเป็น "ขอบ"
                        if 0.3 < aspect_ratio < 3.0:

                            # กรองด้วยความกลมอีกชั้น (Circularity)
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter == 0: continue
                            circularity = 4 * np.pi * (area / (perimeter * perimeter))

                            if circularity > 0.4: # ปรับให้เข้มงวดขึ้น (0.4-0.6)
                                defect_count += 1
                                (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                                cv2.circle(frame, (int(cx + x1), int(cy + y1)), 18, (0, 0, 255), 2)

                # --- [DISPLAY] ---
                status_txt = "NG" if defect_count > 0 else "PASS"
                status_clr = (0, 0, 255) if defect_count > 0 else (0, 255, 0)
                cv2.putText(frame, f"RESULT: {status_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_clr, 2)
                if defect_count > 0:
                    cv2.putText(frame, f"FOUND: {defect_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1) # กรอบ ROI สีฟ้าอ่อน
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                with lock:
                    output_frame = buffer.tobytes()

            if grabResult: grabResult.Release()
    except Exception as e: print(f"Error: {e}")
    finally:
        if 'camera' in locals(): camera.Close()

@app.route('/')
def index(): return '<body style="background:#000; display:flex; justify-content:center;"><img src="/video_feed" style="height:90vh;"></body>'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if output_frame is None: continue
                frame = output_frame
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)