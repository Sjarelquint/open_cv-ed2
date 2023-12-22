import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd='D:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img=cv2.imread("2.png")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# print(pytesseract.image_to_string(img_rgb))
hImg,wImg,_=img_rgb.shape
boxes=pytesseract.image_to_boxes(img_rgb)
for box in boxes.splitlines():
    box=box.split(' ')
    print(box)
    x,y,w,h=int(box[1]),int(box[2]),int(box[3]),int(box[4])
    cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),2)
    cv2.imshow('img',img)
cv2.waitKey(0)