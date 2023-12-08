import cv2
import numpy as np
import mediapipe as mp
import time
from math import hypot

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                               max_num_faces=2,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
currenttime = 0
previoustime = 0


def get_ratio(points, landms):
    h, w, c = frame.shape
    left_point = (int(landms.landmark[points[0]].x * w), int(landms.landmark[points[0]].y * h))
    right_point = (int(landms.landmark[points[1]].x * w), int(landms.landmark[points[1]].y * h))
    upper_point = (int(landms.landmark[points[2]].x * w), int(landms.landmark[points[2]].y * h))
    lower_point = (int(landms.landmark[points[3]].x * w), int(landms.landmark[points[3]].y * h))
    # hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
    # ver_line = cv2.line(frame, upper_point, lower_point, (255, 0, 0), 2)
    ver_line_length = hypot((upper_point[0] - lower_point[0]), (upper_point[1] - lower_point[1]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ratio = hor_line_length / (ver_line_length + 0.00000001)
    return ratio


while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            h, w, c = frame.shape
            left_ratio = get_ratio([33, 133, 159, 145], faceLms)
            right_ratio = get_ratio([362, 263, 386, 374], faceLms)
            ratio = (left_ratio + right_ratio) / 2
            if ratio > 4:
                cv2.putText(frame, 'blink', (50, 150), font, 2, (0, 0, 255), 2)
            left_eye_region = np.array([(faceLms.landmark[33].x * w, faceLms.landmark[33].y * h),
                                        (faceLms.landmark[161].x * w, faceLms.landmark[161].y * h),
                                        (faceLms.landmark[159].x * w, faceLms.landmark[159].y * h),
                                        (faceLms.landmark[157].x * w, faceLms.landmark[157].y * h),
                                        (faceLms.landmark[133].x * w, faceLms.landmark[133].y * h),
                                        (faceLms.landmark[154].x * w, faceLms.landmark[154].y * h),
                                        (faceLms.landmark[145].x * w, faceLms.landmark[145].y * h),
                                        (faceLms.landmark[163].x * w, faceLms.landmark[163].y * h)], np.int32)
            # cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2, )

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [left_eye_region], True, 255, 2, )
            cv2.fillPoly(mask, [left_eye_region], 255)

            min_x = np.min(left_eye_region[:, 0])
            max_x = np.max(left_eye_region[:, 0])
            min_y = np.min(left_eye_region[:, 1])
            max_y = np.max(left_eye_region[:, 1])
            left_eye = frame[min_y:max_y, min_x:max_x]
            gray_left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            _, threshold_eye = cv2.threshold(gray_left_eye, 70, 255, cv2.THRESH_BINARY)
            left = cv2.bitwise_and(frame, frame, mask=mask)
            lefty = left[min_y:max_y, min_x:max_x]
            height, width = threshold_eye.shape
            left_thresh = threshold_eye[0:height, 0:int(width / 2)]
            right_thresh = threshold_eye[0:height, int(width / 2):width]
            left_thresh_white_count = cv2.countNonZero(left_thresh)
            right_thresh_white_count = cv2.countNonZero(right_thresh)

            gaze_ratio=left_thresh_white_count/(right_thresh_white_count+0.00000000001)
            cv2.putText(frame, str(ratio), (50, 150), font, 2, (0, 0, 255), 2)

        #     mpDraw.draw_landmarks(
        #         image=frame,
        #         landmark_list=faceLms,
        #         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_tesselation_style())
        #     mpDraw.draw_landmarks(
        #         image=frame,
        #         landmark_list=faceLms,
        #         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_contours_style())
        #     mpDraw.draw_landmarks(
        #         image=frame,
        #         landmark_list=faceLms,
        #         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        #         landmark_drawing_spec=None,
        #         connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_iris_connections_style())
        # for i in mp.solutions.face_mesh.FACEMESH_IRISES:
        #     print(i)
    # cv2.imshow('eye', left_eye)
    cv2.imshow('frame', frame)
    # cv2.imshow('thres', threshold_eye)
    # cv2.imshow("left", left)
    # cv2.imshow("lefty", lefty)
    cv2.imshow('left_count', left_thresh)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
