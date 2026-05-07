
    
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import random

# =========================
# LOAD MODEL
# =========================
emotion_model = load_model("emotion_model.h5")

emotion_names = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

# =========================
# YOUTUBE SONG DATABASE (NO API NEEDED ✅)
# =========================
songs_db = {
    "Happy": [
        ("Happy - Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Can't Stop The Feeling", "https://www.youtube.com/watch?v=ru0K8uYEZWw"),
        ("On Top Of The World", "https://www.youtube.com/watch?v=w5tWYmIOWGk")
    ],
    "Sad": [
        ("Let Her Go", "https://www.youtube.com/watch?v=RBumgq5yVrA"),
        ("Someone Like You", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("Fix You", "https://www.youtube.com/watch?v=k4V3Mo61fJM")
    ],
    "Angry": [
        ("Stronger", "https://www.youtube.com/watch?v=PsO6ZnUZI0g"),
        ("Believer", "https://www.youtube.com/watch?v=7wtfhZwyrcc")
    ],
    "Neutral": [
        ("Lo-fi Chill", "https://www.youtube.com/watch?v=5qap5aO4i9A"),
        ("Relax Music", "https://www.youtube.com/watch?v=2OEL4P1Rz04")
    ],
    "Surprise": [
        ("Uptown Funk", "https://www.youtube.com/watch?v=OPf0YbXqDm0")
    ],
    "Fear": [
        ("Calm Piano", "https://www.youtube.com/watch?v=1ZYbU82GVz4")
    ],
    "Disgust": [
        ("Chill Beats", "https://www.youtube.com/watch?v=DWcJFNfaw9c")
    ]
}

# =========================
# MEDIAPIPE SETUP
# =========================
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(min_detection_confidence=0.6)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

# =========================
# GESTURE FUNCTION
# =========================
def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    tips = [8, 12, 16, 20]
    fingers = []

    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Thumb
    thumb = 1 if lm[4].x > lm[3].x else 0

    if sum(fingers) == 4:
        return "stop"        # ✋

    if sum(fingers) == 0 and thumb == 0:
        return "fist"        # ✊

    if thumb == 1 and sum(fingers) == 0:
        return "thumbs_up"   # 👍

    if thumb == 0 and lm[4].y > lm[3].y:
        return "thumbs_down" # 👎

    return "unknown"

# =========================
# STREAMLIT UI
# =========================
st.title("🎭 Emotion + ✋ Gesture Controlled 🎵 YouTube Player")

run = st.button("Start System")

frame_window = st.image([])
emotion_text = st.empty()
gesture_text = st.empty()
song_box = st.empty()

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

emotion = ""
songs = []
song_index = 0
paused = False

if run:
    st.success("📸 Detecting Emotion... Hold still")

    # -------- CAPTURE EMOTION ONCE --------
    for i in range(20):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face.process(rgb)

        if result.detections:
            for detection in result.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size != 0:
                    face_img = cv2.resize(face_img, (48,48))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img = face_img / 255.0
                    face_img = np.reshape(face_img, (1,48,48,1))

                    pred = emotion_model.predict(face_img, verbose=0)
                    emotion = emotion_names[np.argmax(pred)]

        frame_window.image(frame, channels="BGR")

    # -------- LOAD SONGS --------
    songs = songs_db.get(emotion, [])
    random.shuffle(songs)

    st.success(f"🎯 Emotion Detected: {emotion}")

    # =========================
    # MAIN LOOP (GESTURE CONTROL)
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gesture = ""

        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                gesture = detect_gesture(hand_landmarks)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # -------- GESTURE ACTIONS --------
        if gesture == "stop":
            paused = True

        elif gesture == "thumbs_up":
            paused = False

        elif gesture == "thumbs_down":
            song_index = (song_index + 1) % len(songs)

        elif gesture == "fist":
            st.warning("🔄 Reset System")
            break

        # -------- DISPLAY --------
        frame_window.image(frame, channels="BGR")

        emotion_text.subheader(f"Emotion: {emotion}")
        gesture_text.subheader(f"Gesture: {gesture}")

        if songs:
            song, url = songs[song_index]

            if not paused:
                song_box.markdown(f"### 🎵 {song}")
                st.video(url)
            else:
                song_box.markdown("⏸️ Paused")

cap.release()




















