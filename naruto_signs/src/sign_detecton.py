import numpy as np
import cv2
import mediapipe as mp
import joblib

svm = joblib.load('naruto_signs/models/svm_model.pkl')
data = np.load('naruto_signs/data/features_train.npz')
labels = data['labels']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 1,
    min_detection_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            wrist = hand.landmark[0]
            hand_coordinates = []
            for i in range(21):
                hand_coordinates.append(hand.landmark[i].x - wrist.x)
                hand_coordinates.append(hand.landmark[i].y - wrist.y)
                hand_coordinates.append(hand.landmark[i].z - wrist.z)
            coordinates = np.array(hand_coordinates).reshape(1,-1)
            sign_name = svm.predict(coordinates)[0]

            cv2.putText(frame,sign_name, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        else:
            cv2.putText(frame, 'No Hands Detected', (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Webcam",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()










