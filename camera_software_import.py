#importing libraries and downloading mediapipe and open cv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = '/absolute/path/to/gesture_recognizer.task'


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()
while(True):
    data,image = cap.read()
    #Flips image
    image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
    #stores results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)
    cv2.imshow('Handtracker', image)
    cv2.waitKey(1)