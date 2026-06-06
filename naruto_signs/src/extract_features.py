import cv2
import os
import mediapipe as mp
import numpy as np
path ='naruto_signs/data/train'

mp_hands = mp.solutions.hands

features = []
labels = []
with mp_hands.Hands(
    static_image_mode= True,
    max_num_hands=2,
    min_detection_confidence = 0.7
) as hands:

    for files in os.listdir(path=path):
        sign_folder = os.path.join(path, files)
        for img_name in os.listdir(sign_folder):
            img = cv2.imread(os.path.join(sign_folder, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img_coordinates = []

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                wrist = hand.landmark[0]
                for i in range(21):
                    img_coordinates.append(hand.landmark[i].x - wrist.x)
                    img_coordinates.append(hand.landmark[i].y - wrist.y)
                    img_coordinates.append(hand.landmark[i].z - wrist.z)
            features.append(img_coordinates)
            labels.append(files)

clean_features = [f for f in features if len(f) == 63]
clean_labels = [labels[i] for i, f in enumerate(features) if len(f) == 63]

print(f"Good vectors: {len(clean_features)} out of {len(features)}")


np.savez("naruto_signs/data/features_train.npz",
        features=np.array(clean_features),
        labels=np.array(clean_labels))




data = np.load("naruto_signs/data/features_train.npz")
print(f"Features shape: {data['features'].shape}")
print(f"Labels shape: {data['labels'].shape}")
print(f"Unique Labels: {np.unique(data['labels'])}")