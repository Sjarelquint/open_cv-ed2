import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
currenttime = 0
previoustime = 0

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # #static pose in 3d
        # mpDraw.plot_landmarks(results.pose_world_landmarks,mpPose.POSE_CONNECTIONS)
        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     h, w, c = frame.shape
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #     print(id, cx, cy)
        #     cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
        h, w, c = frame.shape  # get shape of original frame
        opImg = np.zeros([h, w, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black

        # draw extracted pose on black white image
        mpDraw.draw_landmarks(opImg, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec((255, 0, 0), 2, 2),
                              mpDraw.DrawingSpec((255, 0, 255), 2, 2)
                              )
        # display extracted pose on blank images
        cv2.imshow("Extracted Pose", opImg)

    currenttime = time.time()
    fps = int(1 / (currenttime - previoustime))
    previoustime = currenttime
    cv2.putText(frame, f'FPS {fps}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()
