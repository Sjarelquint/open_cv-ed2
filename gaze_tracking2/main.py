import mediapipe as mp
import cv2
import numpy as np
import math

font = cv2.FONT_HERSHEY_PLAIN
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
R_UPPER = 159
R_LOWER = 145
L_UPPER = 386
L_LOWER = 374
keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}


def Letter(letter_index, text, letter_light):
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400
    width = 200
    height = 200
    th = 3  # thickness
    if letter_light == True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4

    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)


def dist(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def blink_detector(right, left, upper, lower):
    gor_dist = dist(right, left)
    ver_dist = dist(upper, lower)
    blink_ratio = gor_dist / (ver_dist + 0.00000001)
    blink_position = ''
    if blink_ratio > 4:
        blink_position = "blink"
    return blink_position


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
board = np.zeros((500, 500), np.uint8)
board[:] = 255
new_frame = np.zeros((500, 500, 3), np.uint8)
with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    frames = 0
    letter_index = 0
    blinking_frames = 0
    text = ""
    while True:
        ret, frame = cap.read()
        w, h, _ = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keyboard[:] = (0, 0, 0)
        frames += 1
        active_letter = keys_set_1[letter_index]
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
            cv2.putText(frame, f'iris position:{iris_position} ratio:{ratio:.2f}', (30, 30), font, 2, (0, 0, 255), 2)
            blink_position = blink_detector(mesh_points[MOST_R_RIGHT], mesh_points[MOST_L_RIGHT],
                                            mesh_points[R_UPPER], mesh_points[R_LOWER])
            cv2.putText(frame, f'{blink_position}', (60, 60), font, 2, (0, 0, 255), 2)
            if blink_position == 'blink':
                blinking_frames += 1
                frames -= 1
                if blinking_frames == 5:
                    text += active_letter

            else:
                blinking_frames = 0
        if ratio <= 0.42:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif ratio > 0.42 and ratio <= 0.57:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
        if frames == 10:
            letter_index += 1
            frames = 0
        if letter_index == 15:
            letter_index = 0
        for i in range(15):
            if i == letter_index:
                light = True
            else:
                light = False
            Letter(i, keys_set_1[i], light)
        cv2.putText(board, text, (10, 100), font, 4, 0, 3)
        cv2.imshow('frame', frame)
        cv2.imshow('new_frame', new_frame)
        cv2.imshow('keyboard', keyboard)
        cv2.imshow("Board", board)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
