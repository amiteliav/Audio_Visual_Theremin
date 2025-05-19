"""
See reference git repo:
https://github.com/eoinfennessy/webcam-theremin
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

from visual import Camera, is_in_exit_zone, get_pitch_area_bounds, is_in_depth_box_zone
from visual import draw_exit_button, draw_pitch_area, draw_pitch_lines, draw_frequency_text, draw_depth_toggle_box
from hands import HandDetector
from sounds import generate_pitch_dict, ToneGenerator
from depth_estimation import DepthEstimator


def test_hand_tracking():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test_sounds():
    fs = 44100  # Sampling rate
    duration = 1  # seconds
    f = 440.0  # Frequency (Hz)

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    wave = 0.1 * np.sin(2 * np.pi * f * t)

    sd.play(wave, samplerate=fs)
    sd.wait()


# ========================


def main():
    # Define pitch range
    pitch_range = ["C4", "C5"]
    pitch_dict = generate_pitch_dict(*pitch_range)  # *args unpacks the list
    print(f"Pitch range: {pitch_range}")
    print(f"Pitch dictionary: {pitch_dict}")

    # Initialize all the components
    cam = Camera()
    hand_detector = HandDetector()
    tone_gen = ToneGenerator(pitch_range=pitch_range)
    depth_estimator = DepthEstimator(smoothing_alpha=0.95, smooth=True, blur_kernel=(3, 3))

    # Set depth estimation flag
    depth_frame_count = 0
    DEPTH_ESTIMATION_INTERVAL = 6 # Process every N frame
    depth_enabled = False
    depth_box_was_pressed = False

    try:
        while True:
            # Get the frame from the camera
            frame = cam.get_frame()
            frame = cv2.flip(frame, 1)  # Flip horizontally like a mirror

            # Get frame shape and the tracking area
            h, w, _ = frame.shape
            pitch_area = get_pitch_area_bounds(w, h)

            # Get the hand landmarks
            results = hand_detector.detect(frame)
            hand_detector.draw_landmarks(frame, results)

            # Draws over the frame
            draw_exit_button(frame)
            draw_pitch_area(frame)
            draw_pitch_lines(frame, pitch_area, pitch_dict)
            frame = draw_depth_toggle_box(frame, depth_enabled)


            # Get the position of the index fingertip
            pos = hand_detector.get_index_fingertip_position(results, w, h)

            # Check if a hand was detected - and if so, update everything
            if pos:
                x, y = pos  # Unpack the position
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Draw fingertip

                # ========= Check if boxes are pressed =========
                # Exit if the fingertip is in the exit zone
                if is_in_exit_zone(x, y):
                    print("Exit area triggered by hand.")
                    break

                # Switch the Depth Estimation on/off
                in_depth_box = is_in_depth_box_zone(x, y, h, w)
                if in_depth_box and not depth_box_was_pressed:
                    depth_enabled = not depth_enabled
                    depth_box_was_pressed = True  # Prevent retriggering while finger stays inside
                elif not in_depth_box:
                    depth_box_was_pressed = False  # Reset when finger leaves box
                # ===================================================


                # If the depth estimation is enabled, get the depth value and do something
                depth_val = None
                if depth_enabled and (depth_frame_count % DEPTH_ESTIMATION_INTERVAL == 0):
                    if 0 <= x < w and 0 <= y < h:
                        depth_map, depth_norm = depth_estimator.estimate_depth(frame)
                        depth_val = depth_map[y, x]
                        depth_norm_val = depth_norm[y, x]

                        # Prepare the text strings
                        text_depth = f"Depth Raw: {depth_val:.1f} | Depth Norm: {depth_norm_val}"

                        # Calculate positions for centered text at the top
                        (text_w, text_h), _ = cv2.getTextSize(text_depth, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        x_text = (w - text_w) // 2
                        y_text = 30  # gap px from top

                        # Put the text on the frame
                        cv2.putText(frame, text_depth, (x_text, y_text),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                # ------------------

                # Update the pitch based on the x position of the fingertip
                freq = tone_gen.update_pitch_from_xy(x, y, w, pitch_area)
                amplitude = tone_gen.update_amplitude_from_xy(x, y, h, pitch_area)  # we dont use the return value
                tone_gen.depth_to_num_harmonics(depth_val) # update the number of harmonics based on depth

                # Draw the frequency on the frame
                draw_frequency_text(frame, freq)
            # ----------------------

            # Draw the frame
            cv2.imshow("Audio-Visual Theremin", frame)

            # Exit if 'ESC' is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit| 0xFF == 27 the ASCII code for  ESC
                break

    finally:
        cam.release()
        tone_gen.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello World")
    print("-----------------------")

    # # Test the hand tracking functionality
    # print("Testing hand tracking...")
    # test_hand_tracking()

    # # Test the sound functionality
    # print("Testing sound...")
    # test_sounds()

    main()

