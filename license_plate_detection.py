import cv2
import numpy as np
import os
import pytesseract

# path Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ประมวลผลภาพ
def convertImage(image):
    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # เบลอภาพ
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # ใช้ Canny Edge Detection
    canny = cv2.Canny(blur, 100, 200)
    return canny


img = cv2.imread("test.jpg")
processed_img = convertImage(img)
original_img = img.copy()

# คัดลอกภาพเพื่อใช้หาขอบเขตของ contour
contour_img = processed_img.copy()

# หาค่าของ contour
contours, heirarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# สร้างโฟลเดอร์สำหรับบันทึกภาพป้ายทะเบียน (ถ้าไม่มี)
output_folder = "detected_plates"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ตัวนับสำหรับการบันทึกภาพ
plate_saved = False

# ตัวนับจำนวนป้ายทะเบียนที่บันทึก
plate_count = 0

for contour in contours:
    # คำนวณความยาวเส้นรอบรูป
    p = cv2.arcLength(contour, True)
    # หารูปทรงที่มี 4 ด้าน (ประมาณเป็นสี่เหลี่ยม)
    approx = cv2.approxPolyDP(contour, 0.02 * p, True)

    if len(approx) == 4:  # ตรวจสอบว่าเป็นรูปสี่เหลี่ยม
        x, y, w, h = cv2.boundingRect(contour)
        license_img = original_img[y:y+h, x:x+w]

        # บันทึกเฉพาะภาพแรกที่เจอป้ายทะเบียน
        if license_img.size > 0 and not plate_saved:
            save_path = os.path.join(output_folder, f"license_plate_{plate_count+1}.jpg")
            cv2.imwrite(save_path, license_img)
            print(f"บันทึกป้ายทะเบียนที่: {save_path}")
            plate_saved = True  # ตั้งค่าว่าได้บันทึกแล้ว
            plate_count += 1  # เพิ่มจำนวน

            # กำหนด path ของไฟล์รูปป้ายทะเบียน
            input_image_path = f"detected_plates/license_plate_{plate_count}.jpg"

            # ตรวจสอบว่ามีไฟล์รูปป้ายทะเบียนหรือไม่
            if os.path.exists(input_image_path):
                # โหลดรูปภาพป้ายทะเบียน
                license_img = cv2.imread(input_image_path)
                
                # แปลงรูปภาพเป็นสีเทา
                gray_img = cv2.cvtColor(license_img, cv2.COLOR_BGR2GRAY)
                
                # ปรับภาพให้คมชัดขึ้น (ถ้าจำเป็น)
                processed_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # ใช้ pytesseract เพื่อแปลงภาพเป็นข้อความ (ภาษาไทยเท่านั้น)
                text = pytesseract.image_to_string(processed_img, lang='tha', config='--psm 6')
                
                # แสดงผลลัพธ์
                print("ข้อความที่ตรวจพบจากป้ายทะเบียน:")
                print(text)
                
            else:
                print("ไม่พบไฟล์รูปป้ายทะเบียนใน path ที่กำหนด")
            break  # ออกจาก loop หลังจากบันทึกป้ายทะเบียนภาพแรก

# แสดงผลกรอบป้ายทะเบียน & ภาพป้ายทะเบียน
if plate_saved:
    cv2.drawContours(img, [contour], -1, (0, 255, 255), 2)
    cv2.imshow("image", img)
    cv2.imshow("Detected license plate", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ไม่พบป้ายทะเบียนในภาพนี้")
