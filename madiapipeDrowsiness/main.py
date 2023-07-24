import streamlit as st
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import draw_landmarks_on_image
import pickle
import pandas as pd
import numpy as np

with open('drowsiness.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("drowsiness detection")
frame_placeholder = st.empty()

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("The video capture has ended.")
        break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    try:
        face = detection_result.face_landmarks[0]
        points = list(np.array([[landmark.x, landmark.y, landmark.z, ] for landmark in face]).flatten())
        X = pd.DataFrame([points])
        cl = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        cv2.putText(annotated_image, cl, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_image, str(round(prob[np.argmax(prob)], 2)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    except:
        pass

    frame_placeholder.image(annotated_image, channels="BGR")
