import cv2
import pytesseract
import os

# ตั้งค่า path สำหรับ Tesseract (แก้ตาม path ที่คุณติดตั้ง)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# กำหนด path ของไฟล์รูปป้ายทะเบียน
input_image_path = "detected_plates/license_plate_1.jpg"

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
    
    # แสดงภาพที่แปลง
    cv2.imshow("License Plate", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ไม่พบไฟล์รูปป้ายทะเบียนใน path ที่กำหนด")
