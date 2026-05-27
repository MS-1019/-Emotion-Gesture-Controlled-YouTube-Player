import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import pyautogui
from collections import Counter
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Emotion + Gesture Music Player",
    layout="wide"
)

st.title("          🎭 Emotion + ✋ Gesture Controlled 🎵 Music Player        ")

# =====================================================
# LOAD MODEL
# =====================================================
emotion_model = load_model("emo_model.h5")

emotion_names = [
    "Angry",
    "Fear",
    "Happy",
    "Sad",
    "Surprise"
]

# =====================================================
# SONG DATABASE
# =====================================================
songs_db = {

    "Happy": [
        ("Gallan Goodiyaan", "https://www.youtube.com/watch?v=jCEdTq3j-0U&list=RDjCEdTq3j-0U&start_radio=1"),
        ("Rasputin", "https://www.youtube.com/watch?v=x5Oag4hISgU"),
        ("Happy - Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Can't Stop The Feeling", "https://www.youtube.com/watch?v=ru0K8uYEZWw"),
        ("On Top Of The World", "https://www.youtube.com/watch?v=w5tWYmIOWGk"),
        ("Refreshing Playlist", "https://www.youtube.com/watch?v=S04xHs5l93k")
    ],

    "Sad": [
        ("Mann Mera","https://www.youtube.com/watch?v=1ykU4QkchSE"),
        ("Skyfall", "https://www.youtube.com/watch?v=sZrTJesvJeo"),
        ("Fix You", "https://www.youtube.com/watch?v=k4V3Mo61fJM"),
        ("Someone Like You", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("Let Her Go", "https://www.youtube.com/watch?v=RBumgq5yVrA")
    ],

    "Angry": [
        ("Lola Blanc-Angry Too ","https://www.youtube.com/watch?v=MqekZVbtI2Q"),
        ("Jee Karda", "https://www.youtube.com/watch?v=VAJK04HOLd0"),
        ("Arjan Vailly", "https://www.youtube.com/watch?v=zqGW6x_5N0k"),
        ("Believer", "https://www.youtube.com/watch?v=7wtfhZwyrcc"),
        ("Stronger", "https://www.youtube.com/watch?v=PsO6ZnUZI0g")
    ],

    "Fear": [
        ("Fear Song","https://www.youtube.com/watch?v=WRoLW48WOBg"),
        ("Bloody Mary", "https://www.youtube.com/watch?v=MsXdUtlDVhk"),
        ("Running Up That Hill", "https://www.youtube.com/watch?v=2pdkKo-Cj5Y"),
        ("Calm Piano", "https://www.youtube.com/watch?v=1ZYbU82GVz4")
    ],

    "Surprise": [
        ("Saiyaara Reprise","https://www.youtube.com/watch?v=asG7cwxi1sA"),
        ("Sitaare","https://www.youtube.com/watch?v=nDjloeIB3Pc"),
        ("Gehra Hua","https://www.youtube.com/watch?v=GX9x62kFsVU&list=PLKrIrxcLptKzznsv2uw5jpUORfiE8x1aV"),
        ("Uptown Funk", "https://www.youtube.com/watch?v=OPf0YbXqDm0")
    ]
}

# =====================================================
# MEDIAPIPE
# =====================================================
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_detector = mp_face.FaceDetection(
    min_detection_confidence=0.7
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================================
# STABLE GESTURE DETECTION
# =====================================================
gesture_history = []

def detect_gesture(hand_landmarks):

    lm = hand_landmarks.landmark

    fingers = []

    # =====================================
    # INDEX
    # =====================================
    fingers.append(
        1 if lm[8].y < lm[6].y else 0
    )

    # MIDDLE
    fingers.append(
        1 if lm[12].y < lm[10].y else 0
    )

    # RING
    fingers.append(
        1 if lm[16].y < lm[14].y else 0
    )

    # PINKY
    fingers.append(
        1 if lm[20].y < lm[18].y else 0
    )

    # =====================================
    # THUMB
    # =====================================
    thumb_up = lm[4].y < lm[3].y
    thumb_down = lm[4].y > lm[3].y

    # =====================================
    # GESTURES
    # =====================================
    # gesture = "NONE"

    # # ✋ FOUR FINGERS UP (Index + Middle + Ring + Pinky)
    # if fingers == [1,1,1,1]:
    #     gesture = "VOLUME_UP"

    # # ☝ INDEX ONLY
    # elif fingers == [1,0,0,0]:
    #     gesture = "PLAY"

    # # 🤙 PINKY ONLY
    # elif fingers == [0,0,0,1]:
    #     gesture = "VOLUME_DOWN"

    # # 👎 THUMB DOWN
    # elif thumb_up and fingers == [0,0,0,0]:
    #     gesture = "RESTART"

    # ✋ FOUR FINGERS DOWN
    # # (All fingers folded)
    # elif fingers == [0,0,0,0]:
    #     gesture = "VOLUME_DOWN"

    # return gesture
    gesture = "NONE"

    # ✋ OPEN HAND
    if fingers == [1,1,1,1]:
        gesture = "STOP"

    # 👍 THUMB UP
    elif thumb_up and fingers == [0,0,0,0]:
        gesture = "PLAY"

    # 👎 THUMB DOWN
    elif thumb_down and fingers == [0,0,0,0]:
        gesture = "RESTART"

    # ☝ INDEX ONLY
    elif fingers == [1,0,0,0]:
        gesture = "VOLUME_UP"

    # 🤙 PINKY ONLY
    elif fingers == [0,0,0,1]:
        gesture = "VOLUME_DOWN"

    # ✌ PEACE
    elif fingers == [1,1,0,0]:
        gesture = "PAUSE"

    return gesture

# =====================================================
# UI
# =====================================================
start = st.button("▶ Start System")

frame_placeholder = st.image([])

emotion_placeholder = st.empty()

gesture_placeholder = st.empty()

song_placeholder = st.empty()

status_placeholder = st.empty()

instruction_placeholder = st.empty()

# =====================================================
# MAIN APP
# =====================================================
if start:

    cap = cv2.VideoCapture(0)

    detected_emotion = "Happy"

    emotion_votes = []

    songs = []

    current_song_index = 0

    volume_level = 50

    paused = False

    last_action_time = 0

    cooldown = 2

    instruction_placeholder.info(
        # """
        # ☝ PLAY MUSIC

        # 👎 RESTART EMOTION

        # 🤙 STOP SYSTEM

        # ✋ VOLUME UP

        # 👇 VOLUME DOWN

        # NONE = NO ACTION
        # """
        """
        👍 PLAY MUSIC
        
        👎 RESTART EMOTION
        
        ✋ STOP SYSTEM
        
        ☝ VOLUME UP
        
        🤙 VOLUME DOWN
        
        ✌ PAUSE / RESUME
        """
    )

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.error("Camera Error")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =================================================
        # FACE DETECTION
        # =================================================
        face_results = face_detector.process(rgb)

        if face_results.detections:

            for detection in face_results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)

                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1 = max(0, x1)
                y1 = max(0, y1)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size != 0:

                    gray = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2GRAY
                    )

                    gray = cv2.resize(gray, (48,48))

                    gray = gray / 255.0

                    gray = np.reshape(
                        gray,
                        (1,48,48,1)
                    )

                    pred = emotion_model.predict(
                        gray,
                        verbose=0
                    )

                    emotion_index = np.argmax(pred)

                    current_emotion = emotion_names[emotion_index]

                    emotion_votes.append(current_emotion)

                    # Keep only latest predictions
                    if len(emotion_votes) > 20:
                        emotion_votes.pop(0)

                    # Majority voting
                    detected_emotion = Counter(
                        emotion_votes
                    ).most_common(1)[0][0]

                    # DRAW FACE
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        detected_emotion,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )

        # =================================================
        # HAND DETECTION
        # =================================================
        hand_results = hands.process(rgb)

        gesture = "NONE"

        if hand_results.multi_hand_landmarks:

            for hand_landmarks in hand_results.multi_hand_landmarks:

                gesture = detect_gesture(hand_landmarks)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # =================================================
        # GESTURE STABILIZATION
        # =================================================
        gesture_history.append(gesture)

        if len(gesture_history) > 15:
            gesture_history.pop(0)

        stable_gesture = Counter(
            gesture_history
        ).most_common(1)[0][0]

        # =================================================
        # DISPLAY
        # =================================================
        frame_placeholder.image(
            frame,
            channels="BGR"
        )

        emotion_placeholder.subheader(
            f"🎭 Emotion: {detected_emotion}"
        )

        gesture_placeholder.subheader(
            f"✋ Gesture: {stable_gesture}"
        )

        # =================================================
        # COOLDOWN
        # =================================================
        current_time = time.time()

        if current_time - last_action_time > cooldown:

            # =============================================
            # PLAY MUSIC
            # =============================================
            if stable_gesture == "PLAY":

                songs = songs_db.get(
                    detected_emotion,
                    []
                )

                if len(songs) > 0:

                    current_song_index = 0

                    song_name, song_url = songs[current_song_index]

                    song_placeholder.markdown(
                        f"## 🎵 {song_name}"
                    )

                    song_placeholder.video(song_url)

                    status_placeholder.success(
                        "👍 Music Playing"
                    )

                    last_action_time = current_time

            # =============================================
            # PAUSE
            # =============================================
            
            
            elif stable_gesture == "PAUSE":

                pyautogui.press("space")

                paused = not paused

                status_placeholder.info(
                    "⏯ Pause / Resume"
                )

                last_action_time = current_time

            # =============================================
            # VOLUME UP
            # =============================================
            elif stable_gesture == "VOLUME_UP":

                pyautogui.press("volumeup")

                volume_level += 10

                if volume_level > 100:
                    volume_level = 100

                status_placeholder.success(
                    f"🔊 Volume: {volume_level}%"
                )

                last_action_time = current_time

            # =============================================
            # VOLUME DOWN
            # =============================================
            elif stable_gesture == "VOLUME_DOWN":

                pyautogui.press("volumedown")

                volume_level -= 10

                if volume_level < 0:
                    volume_level = 0

                status_placeholder.warning(
                    f"🔉 Volume: {volume_level}%"
                )

                last_action_time = current_time

            # =============================================
            # RESTART
            # =============================================
            elif stable_gesture == "RESTART":

                emotion_votes.clear()

                detected_emotion = "Neutral"

                songs.clear()

                song_placeholder.empty()

                status_placeholder.warning(
                    "🔄 Emotion Restarted"
                )

                last_action_time = current_time

            # =============================================
            # STOP
            # =============================================
            elif stable_gesture == "STOP":

                status_placeholder.error(
                    "✋ System Closed"
                )

                break

    # =====================================================
    # RELEASE
    # =====================================================
    cap.release()

    cv2.destroyAllWindows()

    st.success(
        f"🎯 Final Emotion: {detected_emotion}"
    )





