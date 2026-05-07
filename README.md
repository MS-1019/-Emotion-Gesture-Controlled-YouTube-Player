# -Emotion-Gesture-Controlled-YouTube-Player
🧠Core Idea  The system uses:  🎭 Emotion Detection (Face) Detects mood: Happy, Sad, Angry, Neutral, Surprise, etc. ✋ Hand Gesture Recognition Controls playback (play, pause, next, volume, etc.) 🎵 YouTube Music Player Automatically selects and plays songs/videos based on emotion


#- System Architecture

Webcam Input
     ↓
Face Detection (OpenCV + MediaPipe)
     ↓
Emotion Model (CNN / FER model)
     ↓
Gesture Detection (MediaPipe Hands)
     ↓
Control Engine
     ↓
YouTube Player (pywhatkit / pytube / web browser automation)

🧰 ---- Technologies Used
👁️ Computer Vision
OpenCV
MediaPipe (Face + Hands tracking)
🧠 Deep Learning
TensorFlow / Keras (Emotion CNN model)
Pretrained FER model (optional)
🎮 Gesture Control
MediaPipe Hands
Finger landmark detection
🎵 YouTube Control
pywhatkit
webbrowser
selenium (advanced automation)


🎭 ---- Emotion Detection Module
How it works:
Capture face from webcam
Detect face using OpenCV
Preprocess image (grayscale, resize 48x48)
Feed into CNN model
Output emotion label
Example emotions:
😊 Happy → upbeat songs
😢 Sad → emotional/lofi songs
😡 Angry → energetic music
😐 Neutral → random trending songs
😲 Surprise → pop / trending music
✋ 5. Gesture Control System

Using MediaPipe Hands:

Gesture	Action
✊ Fist	Pause
✋ Open palm	Play
👉 Index finger right	Next song
👈 Index finger left	Previous song
🤏 Pinch up/down	Volume control
