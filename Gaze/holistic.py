import cv2
import numpy as np
import mediapipe as mp
cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                               max_num_faces=2,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:

            mpDraw.draw_landmarks(
                image=frame,
                landmark_list=faceLms,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(
                image=frame,
                landmark_list=faceLms,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mpDraw.draw_landmarks(
                image=frame,
                landmark_list=faceLms,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        # for i in mp.solutions.face_mesh.FACEMESH_IRISES:
        #     print(i)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()