import cv2
import pywt
import numpy as np

# --- ตั้งค่าพารามิเตอร์ ---
PIXEL_THRESHOLD = 15000  # ค่ารวมพิกเซลขาวที่ถือว่าเป็นตำหนิ (ปรับตามที่เคย Calibrate)
MIN_AREA = 500           # ขนาดพิกเซลขั้นต่ำที่จะให้วาดวงกลม/กรอบ
BORDER_MARGIN = 100      # ตัดขอบภาพออก (หลบขอบเตารีด) เพื่อไม่ให้วงผิด

def detect_defects(frame):
    # 1. เตรียมภาพ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # SWT ต้องใช้ขนาดเลขคู่
    gray_res = cv2.resize(gray, (640, 480)) 
    
    try:
        # 2. คำนวณ SWT (เลเวล 1)
        coeffs = pywt.swt2(gray_res.astype(np.float64), 'haar', level=1)
        LL, (LH, HL, HH) = coeffs[0]

        # รวมแรงสั่นสะเทือนของขอบ (Magnitude)
        mag = np.abs(LH) + np.abs(HL) + np.abs(HH)
        mag_norm = np.uint8(np.clip(mag, 0, 255))

        # 3. ทำ Binary และกำจัด Noise
        _, binary = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # สร้าง Mask เพื่อไม่ตรวจจับบริเวณขอบชิ้นงาน (ขอบเตารีด)
        mask = np.zeros_like(binary)
        cv2.rectangle(mask, (BORDER_MARGIN, BORDER_MARGIN), (640-BORDER_MARGIN, 480-BORDER_MARGIN), 255, -1)
        binary_cleaned = cv2.bitwise_and(binary, mask)

        # 4. ค้นหาตำแหน่งตำหนิ (Contours)
        contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_count = 0
        display_frame = cv2.resize(frame, (640, 480))
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                defect_count += 1
                # หาพิกัดตำแหน่งรอย
                x, y, wb, hb = cv2.boundingRect(cnt)
                # วาดกรอบสี่เหลี่ยมระบุตำแหน่ง
                cv2.rectangle(display_frame, (x, y), (x + wb, y + hb), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Defect Area: {int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ตัดสินผลลัพธ์
        status = "DEFECT" if defect_count > 0 else "NORMAL"
        color = (0, 0, 255) if status == "DEFECT" else (0, 255, 0)
        
        cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        return display_frame, binary_cleaned

    except Exception as e:
        print(f"Error: {e}")
        return frame, None

# --- ส่วนหลัก: เปิดกล้อง ---
cap = cv2.VideoCapture(0) # 0 คือกล้องเว็บแคม หรือเปลี่ยนเป็น 1, 2 ตามลำดับกล้อง

print("ระบบกำลังทำงาน... กด 'q' เพื่อปิดโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ประมวลผลภาพเรียลไทม์
    result_frame, bin_view = detect_defects(frame)

    # แสดงผลหน้าจอ
    cv2.imshow("Real-time Defect Detection (SWT)", result_frame)
    
    # ถ้าอยากดูภาพ Binary ควบคู่กัน (เพื่อจูนค่า)
    if bin_view is not None:
        cv2.imshow("SWT Binary Map", bin_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()