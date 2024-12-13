import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import csv

medp_hands    = mp.solutions.hands
hand_detector = medp_hands.Hands(min_detection_confidence = 0.1, min_tracking_confidence = 0.1)
medp_drawing  = mp.solutions.drawing_utils

csv_filename = "ISYARA.csv"

header = []
for i in range(21):
    header.extend([f"x{i}", f"y{i}"])
    
with open(csv_filename, mode = 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    
cap = cv2.VideoCapture(0)

count = 0

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_detected = hand_detector.process(rgb_frame)

    if hand_detected.multi_hand_landmarks:
        for landmarks in hand_detected.multi_hand_landmarks:
            
            medp_drawing.draw_landmarks(frame, landmarks, medp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            if cv2.waitKey(1) & 0xFF == ord(' ') :
                
                with open(csv_filename, mode = 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)
                
                count += 1
                print("Data landmark ke-{} direkam !".format(i))

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
