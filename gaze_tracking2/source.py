import mediapipe as mp
import cv2
import numpy as np

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
            # [print(i.x,i.y) for i in results.multi_face_landmarks[0].landmark]
            mesh_points = np.array([np.multiply([p.x, p.y], [h, w]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            cv2.circle(frame, mesh_points[MOST_R_RIGHT],2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[MOST_L_RIGHT],2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, mesh_points[MOST_R_RIGHT],2, (0, 255, 0), 2, cv2.LINE_AA)

            # print(type(mesh_points[MOST_R_RIGHT][0]))
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
