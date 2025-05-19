import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading

# Parameters
fs = 44100
volume = 0.2
pitch = 440.0
stop_stream = False

# Lock to safely update pitch from webcam thread
pitch_lock = threading.Lock()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Audio callback
phase = 0.0
def audio_callback(outdata, frames, time, status):
    global phase
    t = (np.arange(frames) + phase) / fs
    with pitch_lock:
        freq = pitch
    wave = volume * np.sin(2 * np.pi * freq * t)
    outdata[:] = wave.reshape(-1, 1)
    phase += frames
    phase %= fs  # Avoid floating point overflow

# Start audio stream
stream = sd.OutputStream(callback=audio_callback, samplerate=fs, channels=1, blocksize=1024)
stream.start()

# Webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = frame.shape
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand.landmark[8]
            y = index_finger_tip.y  # Normalized [0, 1]

            with pitch_lock:
                pitch = 220 + (1.0 - y) * 880  # Map y to 220–1100 Hz

            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        cv2.imshow("Theremin - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()

