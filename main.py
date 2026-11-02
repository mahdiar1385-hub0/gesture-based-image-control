
import cv2
import mediapipe as mp
import numpy as np
import math


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CAMERA_INDEX = 0
MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

WINDOW_NAME = "Gesture-Based Image Control"


# --------------------------------------------------
# Hand gesture controller
# --------------------------------------------------

class HandController:
    """Track hand gestures and convert them into control values."""

    def __init__(self):
        self.right_value = 50
        self.left_value = 0

        self.right_mode_index = 0
        self.left_mode_index = 0

        self.right_switch_ready = True
        self.left_switch_ready = True

        self.right_modes = [
            "Brightness",
            "Saturation",
            "Hue"
        ]

        self.left_modes = [
            "Zoom",
            "Portrait"
        ]

    @staticmethod
    def calculate_distance(point_a, point_b):
        """Calculate the distance between two hand landmarks."""

        return math.sqrt(
            (point_a.x - point_b.x) ** 2 +
            (point_a.y - point_b.y) ** 2
        )

    @staticmethod
    def get_control_value(distance):
        """Convert finger distance into a value between 0 and 100."""

        return max(0, min(int(distance * 500), 100))

    def update_hand(self, hand_type, landmarks):
        """Update the selected hand's control value and active mode."""

        thumb = landmarks.landmark[4]
        index = landmarks.landmark[8]

        distance = self.calculate_distance(thumb, index)
        value = self.get_control_value(distance)

        # Lowering the index finger switches to the next mode.
        if hand_type == "Right":

            self.right_value = value

            if index.y > thumb.y:
                if self.right_switch_ready:
                    self.right_mode_index = (
                        self.right_mode_index + 1
                    ) % len(self.right_modes)

                    self.right_switch_ready = False
            else:
                self.right_switch_ready = True

        elif hand_type == "Left":

            self.left_value = value

            if index.y > thumb.y:
                if self.left_switch_ready:
                    self.left_mode_index = (
                        self.left_mode_index + 1
                    ) % len(self.left_modes)

                    self.left_switch_ready = False
            else:
                self.left_switch_ready = True

    def get_current_modes(self):
        """Return the currently selected modes."""

        right_mode = self.right_modes[self.right_mode_index]
        left_mode = self.left_modes[self.left_mode_index]

        return right_mode, left_mode


# --------------------------------------------------
# Computer vision processor
# --------------------------------------------------

class ImageProcessor:
    """Apply visual effects based on hand gesture controls."""

    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation

        self.selfie = self.mp_selfie.SelfieSegmentation(
            model_selection=1
        )

    def adjust_colors(self, frame, mode, value):
        """Modify brightness, saturation, or hue using HSV."""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue, saturation, brightness = cv2.split(hsv)

        adjustment = value - 50

        if mode == "Brightness":

            brightness = np.clip(
                brightness.astype(np.int16) + adjustment,
                0,
                255
            ).astype(np.uint8)

        elif mode == "Saturation":

            saturation = np.clip(
                saturation.astype(np.int16) + adjustment,
                0,
                255
            ).astype(np.uint8)

        elif mode == "Hue":

            hue = (
                hue.astype(np.int16) + adjustment
            ) % 180

            hue = hue.astype(np.uint8)

        adjusted_hsv = cv2.merge([
            hue,
            saturation,
            brightness
        ])

        return cv2.cvtColor(
            adjusted_hsv,
            cv2.COLOR_HSV2BGR
        )

    @staticmethod
    def apply_zoom(frame, value):
        """Zoom into the center of the frame."""

        zoom_factor = 1.0 + value / 100

        height, width = frame.shape[:2]

        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2

        cropped = frame[
            y1:y1 + new_height,
            x1:x1 + new_width
        ]

        return cv2.resize(
            cropped,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    def apply_portrait_effect(self, frame, value):
        """Separate the person from the background and blur the background."""

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.selfie.process(rgb_frame)

        if results.segmentation_mask is None:
            return frame

        # Smooth the segmentation mask to create softer edges.
        mask = np.clip(
            results.segmentation_mask,
            0,
            1
        )

        mask = cv2.GaussianBlur(
            mask,
            (15, 15),
            0
        )

        mask = mask[:, :, np.newaxis]

        # The left hand controls the background blur intensity.
        blur_strength = value / 100

        blur_size = int(5 + blur_strength * 95)

        if blur_size % 2 == 0:
            blur_size += 1

        blurred_background = cv2.GaussianBlur(
            frame,
            (blur_size, blur_size),
            0
        )

        # Combine the original foreground with the blurred background.
        result = (
            frame.astype(np.float32) * mask +
            blurred_background.astype(np.float32) * (1 - mask)
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def close(self):
        """Release the segmentation model."""

        self.selfie.close()


# --------------------------------------------------
# Hand tracking
# --------------------------------------------------

class HandTracker:
    """Detect hands and extract their landmarks."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )

    def process(self, frame):
        """Detect hands in the current frame."""

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        return self.hands.process(rgb_frame)

    def draw_landmarks(self, frame, results):
        """Draw detected hand landmarks and finger connections."""

        if not results.multi_hand_landmarks:
            return frame

        for landmarks in results.multi_hand_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

        return frame

    def draw_control_lines(self, frame, results):
        """Draw lines between the thumb and index finger."""

        if not results.multi_hand_landmarks:
            return frame

        height, width = frame.shape[:2]

        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_type = handedness.classification[0].label

            thumb = landmarks.landmark[4]
            index = landmarks.landmark[8]

            point_a = (
                int(thumb.x * width),
                int(thumb.y * height)
            )

            point_b = (
                int(index.x * width),
                int(index.y * height)
            )

            color = (
                (0, 255, 0)
                if hand_type == "Right"
                else (255, 0, 0)
            )

            cv2.line(
                frame,
                point_a,
                point_b,
                color,
                3
            )

        return frame

    def close(self):
        """Release the hand tracking model."""

        self.hands.close()


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open the camera.")

    controller = HandController()
    processor = ImageProcessor()
    tracker = HandTracker()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Failed to read camera frame.")
                break

            # Mirror the camera image for a natural interaction.
            frame = cv2.flip(frame, 1)

            results = tracker.process(frame)

            right_found = False
            left_found = False

            # Process each detected hand independently.
            if results.multi_hand_landmarks:

                for landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    hand_type = handedness.classification[0].label

                    controller.update_hand(
                        hand_type,
                        landmarks
                    )

                    if hand_type == "Right":
                        right_found = True

                    elif hand_type == "Left":
                        left_found = True

            # Reset switching states when a hand disappears.
            if not right_found:
                controller.right_switch_ready = True

            if not left_found:
                controller.left_switch_ready = True

            right_mode, left_mode = controller.get_current_modes()

            # Apply the right-hand color adjustment first.
            processed = processor.adjust_colors(
                frame,
                right_mode,
                controller.right_value
            )

            # Apply the selected left-hand visual effect.
            if left_mode == "Zoom":

                processed = processor.apply_zoom(
                    processed,
                    controller.left_value
                )

            elif left_mode == "Portrait":

                processed = processor.apply_portrait_effect(
                    processed,
                    controller.left_value
                )

            # Draw tracking information after image processing.
            processed = tracker.draw_landmarks(
                processed,
                results
            )

            processed = tracker.draw_control_lines(
                processed,
                results
            )

            # Display the active modes and control values.
            cv2.putText(
                processed,
                f"Right: {right_mode} | Value: {controller.right_value}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                processed,
                f"Left: {left_mode} | Value: {controller.left_value}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 100, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, processed)

            # Press Q to close the application.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # Release all resources even if an error occurs.
        camera.release()
        tracker.close()
        processor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
