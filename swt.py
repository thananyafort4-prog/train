import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# 1. ตั้งค่าเส้นทาง (Paths)
input_folder = r'D:\Internship\my-project\defeat\reject'
output_base = r'D:\Internship\my-project\defeat\output_swt'

if not os.path.exists(output_base):
    os.makedirs(output_base)

# 2. รายชื่อไฟล์ภาพ
files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.bmp', '.png'))]

for filename in files:
    full_path = os.path.join(input_folder, filename)
    image = cv2.imread(full_path, 0)
    
    if image is None:
        continue
        
    print(f"Processing and Saving: {filename}")
    
    # SWT ต้องการมิติเลขคู่
    h, w = image.shape
    image_resized = cv2.resize(image, (w // 2 * 2, h // 2 * 2))
    
    try:
        # 3. คำนวณ SWT2
        coeffs = pywt.swt2(image_resized.astype(np.float64), 'haar', level=1)
        LL, (LH, HL, HH) = coeffs[0]

        # 4. สร้าง Folder ย่อยสำหรับภาพนี้
        name_only = os.path.splitext(filename)[0]
        save_path = os.path.join(output_base, name_only)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # ฟังก์ชันช่วยแปลงและทำ Binary
        def process_img(img_data):
            abs_img = np.abs(img_data)
            normalized = np.uint8(np.clip(abs_img, 0, 255))
            _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return normalized, binary

        # ประมวลผลแต่ละส่วน
        norm_LH, bin_LH = process_img(LH)
        norm_HL, bin_HL = process_img(HL)
        norm_HH, bin_HH = process_img(HH)
        norm_LL = np.uint8(np.clip(LL, 0, 255))

        # 5. บันทึกไฟล์ภาพแยกชิ้น
        cv2.imwrite(os.path.join(save_path, 'LL_approx.jpg'), norm_LL)
        cv2.imwrite(os.path.join(save_path, 'LH_horizontal.jpg'), norm_LH)
        cv2.imwrite(os.path.join(save_path, 'HL_vertical.jpg'), norm_HL)
        cv2.imwrite(os.path.join(save_path, 'HH_diagonal.jpg'), norm_HH)
        cv2.imwrite(os.path.join(save_path, 'Binary_LH.png'), bin_LH)
        cv2.imwrite(os.path.join(save_path, 'Binary_HL.png'), bin_HL)
        cv2.imwrite(os.path.join(save_path, 'Binary_HH.png'), bin_HH)

        # 6. บันทึกภาพรวม (Summary Plot) สำหรับใช้ส่งงาน
        plt.figure(figsize=(15, 10))
        titles = ['Original', 'LL (Approx)', 'Binary LH (H)', 'Binary HL (V)', 'Binary HH (D)']
        images = [image_resized, norm_LL, bin_LH, bin_HL, bin_HH]
        
        for i in range(5):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
            
        plt.savefig(os.path.join(save_path, 'Summary_Result.png'))
        plt.close() # ปิด plot เพื่อประหยัดแรม

    except Exception as e:
        print(f"Error at {filename}: {e}")

print(f"\n All results saved in: {output_base}")