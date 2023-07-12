import mediapipe as mp
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

import numpy as np
import pandas as pd
import pickle

window = ctk.CTk()
window.title('liftApp')
ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
window.geometry('500x650')
classLabel = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE')
counterLabel = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')
classBox = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ctk.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0')


def reset_counter():
    global counter
    counter = 0


button = ctk.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20),
                       text_color="white", fg_color="blue")
button.place(x=10, y=600)
frame = tk.Frame(height=480, width=480)
frame.place(x=60, y=150)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

with open('lift.pkl', 'rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0,0])
bodylang_class =''


def process():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                              mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    try:
        row = np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row])
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1

    except :
        pass

    imgarr = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, process)
    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)


process()
window.mainloop()
