import os
import cv2
import numpy as np

path = 'key'
images = []
Cls = []
myList = os.listdir(path)
descriptors = []
orb = cv2.SIFT_create()
for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    Cls.append(os.path.splitext(cl)[0])


def findDes(imgs):
    for img in imgs:
        kp, des = orb.detectAndCompute(img, None)
        descriptors.append(des)
    return descriptors


descriptors = findDes(images)


def findMatch(descriptions, im,tres=100):
    kp2, des2, = orb.detectAndCompute(im, None)
    bf = cv2.BFMatcher()
    matchList = []
    value=-1
    try:
        for des in descriptions:
            matches = bf.knnMatch(des, des2, 2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    if len(matchList) != 0:
        if max(matchList) > tres:
            value = matchList.index(max(matchList))
            print(value)
    return value


cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    frameBGR = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    id = findMatch(descriptors, frame)
    if id!=-1:
        cv2.putText(frameBGR,Cls[id],(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    cv2.imshow('img', frameBGR)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()
