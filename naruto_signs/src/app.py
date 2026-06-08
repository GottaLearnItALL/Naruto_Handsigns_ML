import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

# ─── Page Config ───
st.set_page_config(page_title="Naruto Jutsu Detector", page_icon="🔥", layout="wide")

# ─── Load Model ───
@st.cache_resource
def load_model():
    return joblib.load("naruto_signs/models/svm_model.pkl")

@st.cache_resource
def load_mediapipe():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

svm = load_model()
hands = load_mediapipe()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# ─── Jutsu Combos ───
JUTSUS = {
    "🔥 Fireball": {"combo": ["snake", "ram", "dog"], "effect": "fireball"},
    "⚡ Chidori": {"combo": ["ox", "hare", "monkey"], "effect": "chidori"},
    "👥 Shadow Clone": {"combo": ["ram", "snake"], "effect": "shadow_clone"},
}

# ─── Session State ───
if "sign_history" not in st.session_state:
    st.session_state.sign_history = []
if "active_effect" not in st.session_state:
    st.session_state.active_effect = None
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# ─── Effect Functions ───
def fireball_jutsu(frame):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    overlay = frame.copy()
    
    for r_offset, color in [(40, (0, 0, 150)), (15, (0, 100, 255)), (0, (0, 200, 255))]:
        r = max(1, 120 + r_offset + np.random.randint(-10, 10))
        cv2.circle(overlay, (center_x, center_y), r, color, -1)
    
    cv2.circle(overlay, (center_x, center_y), max(1, 40), (200, 255, 255), -1)
    
    for _ in range(30):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.randint(0, 180)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        p_size = max(1, np.random.randint(3, 12))
        color = (0, np.random.randint(80, 200), np.random.randint(200, 255))
        cv2.circle(overlay, (px, py), p_size, color, -1)
    
    for _ in range(15):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = 120 + np.random.randint(30, 80)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        cv2.circle(overlay, (px, py), max(1, np.random.randint(1, 5)), (0, 180, 255), -1)
    
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, "KATON: FIREBALL JUTSU", (w // 2 - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
    return frame


def chidori(frame):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    overlay = frame.copy()
    
    cv2.circle(overlay, (center_x, center_y), 8, (255, 255, 255), -1)
    
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
    
    for _ in range(60):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.randint(10, 300)
        px = int(center_x + dist * np.cos(angle))
        py = int(center_y + dist * np.sin(angle))
        cv2.circle(overlay, (px, py), max(1, np.random.randint(1, 5)), (255, 255, 255), -1)
    
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, "CHIDORI!", (w // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 50), 3)
    return frame


def shadow_clone(frame):
    h, w = frame.shape[:2]
    small_w = w // 3
    small = cv2.resize(frame, (small_w, h))
    result = np.zeros_like(frame)
    result[:, 0:small_w] = small
    result[:, small_w:small_w * 2] = small
    result[:, small_w * 2:small_w * 3] = small
    if small_w * 3 < w:
        result[:, small_w * 3:] = frame[:, small_w * 3:]
    cv2.putText(result, "SHADOW CLONE JUTSU!", (w // 2 - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return result


EFFECT_FUNCS = {
    "fireball": fireball_jutsu,
    "chidori": chidori,
    "shadow_clone": shadow_clone,
}

# ─── Feature Extraction ───
def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, None, result
    
    hand = result.multi_hand_landmarks[0]
    wrist = hand.landmark[0]
    coords = []
    for i in range(21):
        coords.append(hand.landmark[i].x - wrist.x)
        coords.append(hand.landmark[i].y - wrist.y)
        coords.append(hand.landmark[i].z - wrist.z)
    
    return np.array(coords).reshape(1, -1), hand, result


def check_combo(history):
    for name, info in sorted(JUTSUS.items(), key=lambda x: len(x[1]["combo"]), reverse=True):
        combo = info["combo"]
        if len(history) >= len(combo) and history[-len(combo):] == combo:
            return name, info["effect"]
    return None, None


# ─── UI ───
st.markdown("""
# 🍥 Naruto Jutsu Detector
**Perform hand signs to activate jutsus!** Take snapshots of each sign in sequence.
""")

# Sidebar
with st.sidebar:
    st.markdown("## 📜 Jutsu Combos")
    for name, info in JUTSUS.items():
        combo_str = " → ".join(info["combo"])
        st.markdown(f"**{name}**")
        st.code(combo_str)
    
    st.divider()
    st.markdown("## 📊 Model Info")
    st.markdown("- **Model:** SVM (RBF kernel, C=100)")
    st.markdown("- **Accuracy:** 96.9%")
    st.markdown("- **Features:** 63-dim hand landmarks")
    st.markdown("- **Classes:** 13 hand signs")
    
    st.divider()
    if st.button("🔄 Reset Combo", use_container_width=True):
        st.session_state.sign_history = []
        st.session_state.active_effect = None
        st.rerun()

# Sign history display
if st.session_state.sign_history:
    cols = st.columns(len(st.session_state.sign_history))
    for i, sign in enumerate(st.session_state.sign_history):
        cols[i].markdown(f"**{i+1}.** `{sign}`")

# Main area
col_cam, col_result = st.columns(2)

with col_cam:
    st.markdown("### 📸 Show your hand sign")
    img_data = st.camera_input("Take a snapshot", key="camera")

if img_data is not None:
    # Decode image
    file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.last_frame = frame.copy()
    
    # Extract landmarks
    features, hand_landmarks, mp_results = extract_landmarks(frame)
    
    # Draw hand skeleton
    if hand_landmarks is not None:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    with col_result:
        if features is not None:
            # Predict
            sign_name = svm.predict(features)[0]
            
            # Add to history if different from last
            if (len(st.session_state.sign_history) == 0 or 
                sign_name != st.session_state.sign_history[-1]):
                st.session_state.sign_history.append(sign_name)
            
            st.markdown(f"### Detected: **{sign_name.upper()}**")
            
            # Draw sign name on frame
            cv2.putText(frame, sign_name.upper(), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Check for jutsu combo
            jutsu_name, effect_key = check_combo(st.session_state.sign_history)
            
            if jutsu_name:
                st.markdown(f"## 🔥 {jutsu_name} ACTIVATED!")
                st.balloons()
                
                # Apply effect
                effect_frame = EFFECT_FUNCS[effect_key](st.session_state.last_frame)
                effect_frame_rgb = cv2.cvtColor(effect_frame, cv2.COLOR_BGR2RGB)
                st.image(effect_frame_rgb, caption=jutsu_name, use_container_width=True)
                
                # Reset history
                st.session_state.sign_history = []
            
            # Show annotated frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Hand landmarks", use_container_width=True)
        else:
            st.warning("No hand detected. Try again with your hand clearly visible.")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
Built with MediaPipe, scikit-learn, and OpenCV | 
Hand sign classification using SVM with 96.9% accuracy
</div>
""", unsafe_allow_html=True)