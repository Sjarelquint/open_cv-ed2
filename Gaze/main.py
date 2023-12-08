import cv2
import numpy as np
import mediapipe as mp
import time
from math import hypot

font=cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                               max_num_faces=2,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(color=(0,255,0),circle_radius=3,thickness=2)
currenttime = 0
previoustime = 0
while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            h, w, c = frame.shape
            left_point = (int(faceLms.landmark[33].x * w), int(faceLms.landmark[33].y * h))
            right_point = (int(faceLms.landmark[133].x * w), int(faceLms.landmark[133].y * h))
            upper_point = (int(faceLms.landmark[159].x * w), int(faceLms.landmark[159].y * h))
            lower_point = (int(faceLms.landmark[145].x * w), int(faceLms.landmark[145].y * h))
            # hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
            # ver_line = cv2.line(frame, upper_point, lower_point, (255, 0, 0), 2)
            ver_line_length = hypot((upper_point[0] - lower_point[0]), (upper_point[1] - lower_point[1]))
            hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
            ratio = hor_line_length/(ver_line_length+0.0000001)
            # print(faceLms.landmark[33].z)
            if ratio > 4:
                cv2.putText(frame, 'blink', (50, 150), font, 2, (0, 0, 255), 2)

            mpDraw.draw_landmarks(
                image=frame,
                landmark_list=faceLms,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            # mpDraw.draw_landmarks(
            #     image=frame,
            #     landmark_list=faceLms,
            #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            # mpDraw.draw_landmarks(
            #     image=frame,
            #     landmark_list=faceLms,
            #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())
        # for i in mp.solutions.face_mesh.FACEMESH_IRISES:
        #     print(i)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
