import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands=1,
                 detection_confidence=0.7, tracking_confidence=0.6,
                 smoothing=0.3,
                 ):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # For smoothing fingertip coordinates
        self.prev_x = None
        self.prev_y = None
        self.alpha = smoothing  # 0.0 = no smoothing, 1.0 = very slow smoothing

    def detect(self, frame):
        """
        Detects hands in the given frame and returns the results.
        :param frame: The frame to process.
        :return: The results of the hand detection.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def get_index_fingertip_position(self, results, frame_width, frame_height):
        """
        Returns the (x, y) position of the index fingertip (landmark 8).
        """
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]  # taking the first hand detected
            index_tip = hand.landmark[8]  # index fingertip landmark
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)

            # Apply smoothing
            if self.prev_x is None or self.prev_y is None:
                self.prev_x = x
                self.prev_y = y
            else:  # Update the position with smoothing
                self.prev_x = int(self.alpha * self.prev_x + (1 - self.alpha) * x)
                self.prev_y = int(self.alpha * self.prev_y + (1 - self.alpha) * y)

            return self.prev_x, self.prev_y  # Return smoothed coordinates
        return None  # Return None if no hand is detected


if __name__ == "__main__":
    print("Hello World")
    print("-----------------------")
