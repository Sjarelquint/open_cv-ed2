import mediapipe as mp
import cv2
import numpy as np
import math
font=cv2.FONT_HERSHEY_PLAIN
# left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# irises Indices list
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOST_R_RIGHT = 33
MOST_L_RIGHT = 133
MOST_R_LEFT = 362
MOST_L_LEFT = 263


def dist(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def irisPosition(center, right, left):
    iris_dist = dist(center, right)
    total_dist = dist(right, left)
    ratio = iris_dist / total_dist
    iris_position = ""
    if ratio <= 0.42:
        iris_position = "right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position = 'center'
    else:
        iris_position = 'left'
    return iris_position, ratio


cap = cv2.VideoCapture(0)
with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        w, h, _ = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frameRGB)
        if results.multi_face_landmarks:
            # for i in results.multi_face_landmarks:
            #     print(i)
            # [print(i) for i in results.multi_face_landmarks[0].landmark]
            mesh_points = np.array([np.multiply([p.x, p.y], [h, w]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            # cv2.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[MOST_R_RIGHT], 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[MOST_L_RIGHT], 2, (0, 255, 0), 2, cv2.LINE_AA)
            iris_position, ratio = irisPosition(center_right, mesh_points[MOST_R_RIGHT], mesh_points[MOST_L_RIGHT])
            # print(iris_position)
            # print(ratio)
            # print(mesh_points)
            cv2.putText(frame, f'iris position:{iris_position} ratio:{ratio:.2f}', (20, 20), font, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
