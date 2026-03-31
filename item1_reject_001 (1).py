import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. แปลงภาพเป็นขาวดำเพื่อหาตำหนิ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. แยกส่วนที่เป็นตำหนิ (สมมติว่าตำหนิคือส่วนที่มืดกว่าค่า 100)
    _, defect_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. คำนวณพื้นที่ (นับจำนวนพิกเซล)
    total_pixels = frame.shape[0] * frame.shape[1]
    defect_pixels = cv2.countNonZero(defect_mask)
    defect_percent = (defect_pixels / total_pixels) * 100

    # 4. ตัดสินผล
    status = "FAIL" if defect_percent > 5.0 else "PASS" # เกณฑ์ 5%
    color = (0, 0, 255) if status == "FAIL" else (0, 255, 0)

    # แสดงผล
    cv2.putText(frame, f"Defect: {defect_percent:.2f}%", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Result: {status}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow("Defect Detection Test", frame)
    cv2.imshow("Mask (What AI sees)", defect_mask) # ดูว่ามันจับตรงไหนเป็นตำหนิ

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()