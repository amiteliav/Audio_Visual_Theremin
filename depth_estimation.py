import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from visual import Camera, is_in_exit_zone, get_pitch_area_bounds
from visual import draw_exit_button, draw_pitch_area, draw_pitch_lines, draw_frequency_text
from hands import HandDetector
from sounds import generate_pitch_dict, ToneGenerator


def example_single_img_depth_estimation():
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)

    # # Chose a model type
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move the model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()  # Set the model to evaluation mode

    # Load the transform to pre-process the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Load the image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Inference
    with torch.no_grad():
        prediction = midas(input_batch)  # Get the prediction

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy and move to cpu
    output = prediction.cpu().numpy()

    # Print the original img shape, and show it
    print(f"img shape:{img.shape}")
    plt.imshow(img)
    plt.show()

    # Print the output shape, type, min and max values, and show the output
    print(f"Output shape:{output.shape}")
    print(f"Output type:{type(output)}")
    print(f"Output type:{output.dtype}")
    print(f"Output min and max:{np.min(output)}, {np.max(output)}")
    plt.imshow(output)
    plt.show()


def process_frame(frame, midas, transform, device):
    """Processes a single frame to generate a depth map."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    print(f"img shape:{img_rgb.shape}")
    print(f"depth map shape:{prediction.shape}")

    depth_map = prediction.cpu().numpy()
    return img_rgb, depth_map


# --------------
class DepthEstimator:
    def __init__(self, model_type="MiDaS_small",
                 smoothing_alpha=0.95,
                 smooth=True, blur_kernel=None):
        """

        :param model_type:
        :param smoothing_alpha: higher -> fast change
        :param smooth:
        :param blur_kernel:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.prev_depth_map = None
        self.alpha = smoothing_alpha
        self.smooth = smooth  # apply temporal smoothing
        self.blur_kernel = blur_kernel  # apply spatial smoothing (Gaussian blur)

    def estimate_depth(self, frame):
        """
        Takes an RGB image and returns (a normalized if true) depth map (0–255 uint8).
        Optionally applies smoothing (in time and in space).
        """
        img_input = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.model(img_input) # Get the prediction

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()  # Convert to numpy and move to cpu

        # === Spatial Blur/Smoothing ===
        if self.blur_kernel:
            depth_map = cv2.GaussianBlur(depth_map, ksize=self.blur_kernel, sigmaX=0.4)

        # === Temporal Smoothing ===
        if self.smooth and self.prev_depth_map is not None:
            depth_map = self.alpha * depth_map + (1 - self.alpha) * self.prev_depth_map

        # Save current raw for next step
        self.prev_depth_map = depth_map.copy()

        # === Normalize to 0–255 for visualization ===
        # ------- Note that we don't save the norm depth for the next step
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        return depth_map, depth_norm


def example_webcam_stream_depth_estimation():
    """
    Opens webcam, runs depth estimation, and shows live RGB + depth side by side.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not accessible")
        return

    estimator = DepthEstimator(smoothing_alpha=0.95,
                               smooth=True, blur_kernel=(3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map, depth_norm = estimator.estimate_depth(rgb)

        # Stack RGB and depth map horizontally
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)  # Apply a color map for better visualization
        combined = np.hstack((frame, depth_colored))

        cv2.imshow("RGB + Depth", combined)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello World")
    print("-----------------------")
    print(f'{torch.cuda.is_available()=}')


    # # ---------
    # example_single_img_depth_estimation()

    # ------
    example_webcam_stream_depth_estimation()
