"""
see: https://pytorch.org/hub/intelisl_midas_v2/
for the depth estimation, and my colab file example
"""
import cv2

# Exit button area (global)
EXIT_BOX_TOP_LEFT = (3, 3)
EXIT_BOX_BOTTOM_RIGHT = (50, 30)
BOX_W, BOX_H = 100, 30

def is_in_exit_zone(x, y):
    x1, y1 = EXIT_BOX_TOP_LEFT
    x2, y2 = EXIT_BOX_BOTTOM_RIGHT
    return x1 <= x <= x2 and y1 <= y <= y2

def is_in_depth_box_zone(x, y, frame_h, frame_w):
    """
    The depth box in the lower left corner of the frame.
    """
    depth_toggle_box = {
        "x": frame_w - BOX_W - 5,  # 5px margin from right
        "y": 5,
        "w": BOX_W,
        "h": BOX_H}

    if (depth_toggle_box["x"] <= x <= depth_toggle_box["x"] + depth_toggle_box["w"] and
            depth_toggle_box["y"] <= y <= depth_toggle_box["y"] + depth_toggle_box["h"]):
        return True
    return False

def get_pitch_area_bounds(frame_width, frame_height):
    x_start = int(0.05 * frame_width)
    x_end = int(0.95 * frame_width)
    y_start = int(0.15 * frame_height)
    y_end = int(0.95 * frame_height)

    pitch_area = {
        'x_start': x_start,
        'x_end': x_end,
        'y_start': y_start,
        'y_end': y_end
    }
    return pitch_area

def draw_exit_button(frame):
    top_left     = EXIT_BOX_TOP_LEFT
    bottom_right = EXIT_BOX_BOTTOM_RIGHT
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

    # Get text size
    text = "Exit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)

    # Calculate centered position
    box_w = bottom_right[0] - top_left[0]
    box_h = bottom_right[1] - top_left[1]
    text_x = top_left[0] + (box_w - text_size[0]) // 2
    text_y = top_left[1] + (box_h + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 0, 255), thickness)


def draw_pitch_lines(frame, pitch_area, pitch_dict):
    # Get the positions of the detection box
    x_start = pitch_area['x_start']
    x_end   = pitch_area['x_end']
    y_start = pitch_area['y_start']
    y_end   = pitch_area['y_end']

    # Get the pitch names and number of pitches
    pitches = list(pitch_dict.keys())
    num_pitches = len(pitches)

    pitch_width = x_end - x_start
    spacing = pitch_width // (num_pitches - 1) if num_pitches > 1 else 0

    for i, pitch in enumerate(pitches):
        x = x_start + i * spacing
        color = (255, 0, 0)

        # Draw pitch label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 2
        text_size, _ = cv2.getTextSize(pitch, font, font_scale, thickness)
        text_x = x - text_size[0] // 2
        text_y = y_start - 10  # Slightly above the pitch area

        cv2.putText(frame, pitch, (text_x, text_y), font, font_scale, color, thickness)

        # Draw vertical pitch line inside the pitch area
        cv2.line(frame, (x, y_start), (x, y_end), color, 1)


def draw_pitch_area(frame):
    h, w = frame.shape[:2]
    pitch_area = get_pitch_area_bounds(w, h)
    cv2.rectangle(frame, (pitch_area['x_start'], pitch_area['y_start']),
                  (pitch_area['x_end'], pitch_area['y_end']), color=(0, 255, 0), thickness=2)


def draw_frequency_text(frame, freq, position=(55, 25), color=(255, 255, 255)):
    """
    Draws the frequency text on the frame.

    Args:
        frame: The image frame from the camera.
        freq: The frequency value to display.
        position: (x, y) position of the text.
        color: Color of the text in BGR.
    """
    if freq is None:
        freq_text = "N/A"
    else:
        freq_text = f"F={freq:.1f}[Hz]"
    cv2.putText(frame, freq_text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def draw_depth_toggle_box(frame, enabled):
    frame_h, frame_w = frame.shape[:2]
    box = {
        "x": frame_w - BOX_W - 5,  # 5px margin from right
        "y": 5,
        "w": BOX_W,
        "h": BOX_H}

    color = (0, 255, 0) if enabled else (0, 0, 255)
    label = "Depth (On)" if enabled else "Depth (Off)"

    # Draw rectangle
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)

    # Draw label
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return frame

# =============

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera-reported FPS: {self.fps}")


    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:  # ret is bool - if the frame was grabbed successfully
            raise IOError("Failed to grab frame")
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello World")
    print("-----------------------")