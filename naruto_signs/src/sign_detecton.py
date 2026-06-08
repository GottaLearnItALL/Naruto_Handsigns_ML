import numpy as np
import cv2
import mediapipe as mp
import joblib
import warnings

warnings.filterwarnings('ignore')
svm = joblib.load('naruto_signs/models/svm_model.pkl')
data = np.load('naruto_signs/data/features_train.npz')
labels = data['labels']
cap = cv2.VideoCapture(0)

sign_history = []
jutsus = {
    'fireball': ['snake', 'ram', 'dog'],#, 'boar', 'horse', 'tiger'],
    'chidori': ['ox', 'hare', 'monkey'],
    'shadow_clone': ['ram', 'snake']
}

active_effect = None
frame_counter = 0
last_sign = None
stable_count = 0
STABLE_THRESHOLD = 10 
hand_pos = (0, 0)
cooldown = 0 



mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation

# Fireball Function
def fireball_jutsu(frame, frame_counter, pos):
    center_x, center_y = pos
    overlay = frame.copy()
    
    # Grows fast at first, then slows down
    base_radius = min(frame_counter * 6, 180)
    r = max(1, base_radius + np.random.randint(-10, 10))
    
    # Outer glow — dark red, largest
    cv2.circle(overlay, (center_x, center_y), r + 40, (0, 0, 150), -1)
    
    # Middle fire — orange
    cv2.circle(overlay, (center_x, center_y), r + 15, (0, 100, 255), -1)
    
    # Inner core — yellow
    cv2.circle(overlay, (center_x, center_y), r, (0, 200, 255), -1)
    
    # Hot center — white
    cv2.circle(overlay, (center_x, center_y), max(1, r // 3), (200, 255, 255), -1)
    
    # Fire particles scattered around
    for _ in range(25):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.randint(0, r + 60)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        p_size = max(1, np.random.randint(3, 12))
        color = (0, np.random.randint(80, 200), np.random.randint(200, 255))
        cv2.circle(overlay, (px, py), p_size, color, -1)
    
    # Ember trails — small particles flying outward
    for _ in range(10):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = r + np.random.randint(30, 80)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        cv2.circle(overlay, (px, py), max(1, np.random.randint(1, 5)), (0, 180, 255), -1)
    
    # Blend
    opacity = max(0.3, 0.6 - frame_counter * 0.001)  # slowly fades over time
    frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)
    
    # Jutsu name
    cv2.putText(frame, "KATON: FIREBALL JUTSU", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    
    return frame


#Chidori function
def chidori(frame, frame_counter, pos):
    center_x, center_y = pos
    overlay = frame.copy()
    
    # Tiny bright core — just a dot
    cv2.circle(overlay, (center_x, center_y), 8, (255, 255, 255), -1)
    
    # Tons of lightning bolts
    for _ in range(20):
        angle = np.random.uniform(0, 2 * np.pi)
        x, y = float(center_x), float(center_y)
        for seg in range(np.random.randint(12, 25)):
            step = np.random.randint(20, 60)
            nx = x + step * np.cos(angle) + np.random.randint(-35, 35)
            ny = y + step * np.sin(angle) + np.random.randint(-35, 35)
            thickness = max(1, np.random.randint(1, 4))
            cv2.line(overlay, (int(x), int(y)), (int(nx), int(ny)),
                     (255, 255, np.random.randint(200, 255)), thickness)
            x, y = nx, ny
    
    # Lots of scattered sparks
    for _ in range(60):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.randint(10, 300)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        cv2.circle(overlay, (px, py), max(1, np.random.randint(1, 5)), (255, 255, 255), -1)
    
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, "CHIDORI!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 50), 3)
    return frame


def shadow_clone(frame, frame_counter, mask):
    h, w = frame.shape[:2]
    
    if frame_counter < 15:
        result = frame.copy()
        for _ in range(60):
            px = np.random.randint(0, w)
            py = np.random.randint(0, h)
            cv2.circle(result, (px, py), np.random.randint(15, 50),
                       (np.random.randint(200, 255),) * 3, -1)
        cv2.putText(result, "SHADOW CLONE JUTSU!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return result
    
    result = frame.copy()
    
    # Extract just you using the mask
    person_only = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Clone positions: x_shift in pixels
    shifts = shifts = [(-500, 0), (-300, -60), (300, -60), (500, 0)]

    for x_shift, y_shift in shifts:
        clone = np.zeros_like(frame)
        clone_mask = np.zeros_like(mask)
        
        # Handle x shift
        if x_shift > 0:
            src_x = slice(0, w - x_shift)
            dst_x = slice(x_shift, w)
        else:
            src_x = slice(-x_shift, w)
            dst_x = slice(0, w + x_shift)
        
        # Handle y shift
        if y_shift > 0:
            src_y = slice(0, h - y_shift)
            dst_y = slice(y_shift, h)
        else:
            src_y = slice(-y_shift, h)
            dst_y = slice(0, h + y_shift)
        
        clone[dst_y, dst_x] = person_only[src_y, src_x]
        clone_mask[dst_y, dst_x] = mask[src_y, src_x]
        
        result = np.where(clone_mask[:, :, None] > 0, clone, result)
    
    cv2.putText(result, "SHADOW CLONE JUTSU!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return result

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 1,
    min_detection_confidence=0.7
) as hands:
    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_results = selfie_seg.process(rgb_frame)
        seg_mask = (seg_results.segmentation_mask > 0.5).astype(np.uint8) * 255
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
            if cooldown > 0:
                cooldown -= 1
            if sign_name == last_sign:
                stable_count += 1
            else:
                stable_count = 0
                last_sign = sign_name
            
            if stable_count == STABLE_THRESHOLD:
                if len(sign_history) == 0 or sign_name != sign_history[-1]:
                    sign_history.append(sign_name)
                    print(f"Registered: {sign_name}")
            
                    for k,v in jutsus.items():
                        seq_length = len(v)
                        if sign_history[-seq_length:] == v and cooldown == 0:
                            active_effect = k
                            frame_counter = 0
                            cooldown = 30  # ignore signs for 30 frames after activation
                            sign_history.clear()
                            break
                                            
            cv2.putText(frame,sign_name, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        else:
            cv2.putText(frame, 'No Hands Detected', (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        if active_effect is not None:
            frame_counter += 1
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                hand_x = int(hand.landmark[9].x * frame.shape[1])
                hand_y = int(hand.landmark[9].y * frame.shape[0])
                hand_pos = (hand_x, hand_y)
            
            if active_effect == 'fireball':
                frame = fireball_jutsu(frame, frame_counter, hand_pos)
            elif active_effect == 'chidori':
                frame = chidori(frame, frame_counter, hand_pos)
            elif active_effect == 'shadow_clone':
                frame = shadow_clone(frame, frame_counter, seg_mask)
            
            if frame_counter > 60:
                active_effect = None
        cv2.imshow("Webcam",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()










