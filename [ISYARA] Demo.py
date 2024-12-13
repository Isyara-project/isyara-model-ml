import cv2
import mediapipe as medp
import tensorflow as tf
import numpy as np
import os

medp_hands    = medp.solutions.hands
hand_detector = medp_hands.Hands(min_detection_confidence = 0.2, min_tracking_confidence  = 0.2)
medp_drawing  = medp.solutions.drawing_utils
script_path   = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(f'{script_path}/model/ISYARA.h5')

cap = cv2.VideoCapture(0)

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
            
            hand_landmarks = []
            
            for landmark in landmarks.landmark :
                x_axis = landmark.x
                y_axis = landmark.y
                
                hand_landmarks.append([x_axis, y_axis])
                
            hand_landmarks = np.reshape(np.array(hand_landmarks), (1, -1))
            input_tensor   = tf.convert_to_tensor(hand_landmarks, dtype = tf.float32)
            
            sign_prediction = model.predict(input_tensor)
            predicted_class = np.argmax(sign_prediction)

            cv2.putText(frame, f'Prediksi : {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
