import cv2
import mediapipe as mp
import numpy as np
import time
import os
# ---------------- SETTINGS ----------------
EAR_THRESHOLD = 0.1835
DOT_DASH_THRESHOLD = 0.5
LETTER_GAP = 1.5
CAMERA_INDEX = 0
PANEL_WIDTH = 290
WINDOW_NAME = "Live Morse Blink Detector"
# ---------------- MORSE CODE ----------------
MORSE_CODE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",
    "E": ".",     "F": "..-.",  "G": "--.",   "H": "....",
    "I": "..",    "J": ".---",  "K": "-.-",   "L": ".-..",
    "M": "--",    "N": "-.",    "O": "---",   "P": ".--.",
    "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",
    "Y": "-.--",  "Z": "--..",

    "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..",
    "9": "----."
}
REVERSE_MORSE = {
    code: letter for letter, code in MORSE_CODE.items()
}
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# ---------------- DETECTOR STATE ----------------
eyes_closed = False
blink_start_time = 0.0
last_blink_time = time.time()
current_letter = ""
decoded_text = ""
status = "Ready - look at the camera"
# The button position is updated each frame.
button_bounds = {"x": 0, "y": 15, "w": 105, "h": 40}
# ---------------- RESET ----------------
def reset_detector():
    global current_letter, decoded_text, eyes_closed
    global blink_start_time, last_blink_time, status
    current_letter = ""
    decoded_text = ""
    eyes_closed = False
    blink_start_time = 0.0
    last_blink_time = time.time()
    status = "Reset - ready for new message"
    print("Detector reset. Ready for a new message.")
def handle_mouse(event, x, y, flags, param):
    # This function only handles clicks on RESET.
    if event == cv2.EVENT_LBUTTONDOWN:
        bounds = param
        if (
            bounds["x"] <= x <= bounds["x"] + bounds["w"]
            and bounds["y"] <= y <= bounds["y"] + bounds["h"]
        ):
            reset_detector()
# ---------------- EYE ASPECT RATIO ----------------
def eye_aspect_ratio(landmarks, indices, width, height):
    points = []
    for index in indices:
        landmark = landmarks[index]
        points.append([
            landmark.x * width,
            landmark.y * height
        ])
    points = np.array(points, dtype=np.float32)
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
# ---------------- MORSE REFERENCE PANEL ----------------
def draw_morse_panel(height, current_sequence, decoded_text, status):
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    cv2.putText(
        panel, "MORSE REFERENCE",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 220, 255), 2
    )
    cv2.putText(
        panel, "Letter  Code",
        (15, 57), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (200, 200, 200), 1
    )
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # Two columns: A-M and N-Z
    for i, letter in enumerate(letters):
        if i < 13:
            x = 15
            row = i
        else:
            x = 155
            row = i - 13

        y = 82 + row * 22
        code = MORSE_CODE[letter]
        if current_sequence and code == current_sequence:
            cv2.rectangle(
                panel,
                (x - 5, y - 16),
                (x + 125, y + 4),
                (65, 90, 35),
                -1
            )
            color = (100, 255, 100)
        else:
            color = (235, 235, 235)

        cv2.putText(
            panel, f"{letter}  {code}",
            (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, color, 1
        )
    # Information at the bottom
    bottom_y = min(385, height - 90)
    bottom_y = max(bottom_y, 365)
    cv2.line(
        panel, (12, bottom_y),
        (PANEL_WIDTH - 12, bottom_y),
        (100, 100, 100), 1
    )
    cv2.putText(
        panel, "CURRENT SEQUENCE:",
        (15, bottom_y + 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (0, 220, 255), 1
    )
    sequence_display = current_sequence or "-"
    cv2.putText(
        panel, sequence_display,
        (15, bottom_y + 49), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (100, 255, 100), 2
    )
    cv2.putText(
        panel, "MESSAGE:",
        (15, bottom_y + 72), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (0, 220, 255), 1
    )
    message_display = decoded_text[-18:] if decoded_text else "-"
    cv2.putText(
        panel, message_display,
        (15, min(bottom_y + 95, height - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1
    )
    return panel
# ---------------- LOAD MEDIAPIPE MODEL ----------------
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker.task"
)
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "face_landmarker.task was not found. "
        "Put it in the same folder as morse_blink.py."
    )
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)
# ---------------- START CAMERA ----------------
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check CAMERA_INDEX or close other camera apps."
    )
# Create the window and register the mouse callback ONCE.
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(
    WINDOW_NAME,
    handle_mouse,
    button_bounds
)
# ---------------- MAIN LOOP ----------------
start_time = time.time()
last_timestamp_ms = 0
try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read from webcam.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.time()
            # Convert frame for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            timestamp_ms = int((now - start_time) * 1000)
            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
            # ---------------- DETECT BLINKS ----------------
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                left_ear = eye_aspect_ratio(
                    landmarks, LEFT_EYE, width, height
                )
                right_ear = eye_aspect_ratio(
                    landmarks, RIGHT_EYE, width, height
                )
                ear = (left_ear + right_ear) / 2.0
                cv2.putText(
                    frame, f"EAR: {ear:.3f}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )
                if ear < EAR_THRESHOLD:
                    if not eyes_closed:
                        eyes_closed = True
                        blink_start_time = now
                    closed_duration = now - blink_start_time
                    if closed_duration >= DOT_DASH_THRESHOLD:
                        status = "Eyes closed: DASH (-)"
                    else:
                        status = "Eyes closed: DOT (.)"
                elif eyes_closed:
                    closed_duration = now - blink_start_time
                    if closed_duration < DOT_DASH_THRESHOLD:
                        current_letter += "."
                        status = "DOT recorded (.)"
                    else:
                        current_letter += "-"
                        status = "DASH recorded (-)"
                    eyes_closed = False
                    last_blink_time = now
                cv2.putText(
                    frame,
                    f"Sequence: {current_letter or '-'}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Status: {status}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 2
                )
            else:
                status = "No face detected"
                cv2.putText(
                    frame, status,
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )
            # ---------------- DECODE LETTER ----------------
            if (
                current_letter
                and not eyes_closed
                and now - last_blink_time >= LETTER_GAP
            ):
                letter = REVERSE_MORSE.get(current_letter, "?")
                decoded_text += letter
                print(f"Morse: {current_letter} -> {letter}")
                current_letter = ""
                status = f"Decoded: {letter}"
            # Display message on the camera feed
            cv2.putText(
                frame, f"Message: {decoded_text[-25:]}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2
            )
            # ---------------- BUILD DISPLAY ----------------
            panel = draw_morse_panel(
                height,
                current_letter,
                decoded_text,
                status
            )
            # IMPORTANT: create combined_frame BEFORE using it.
            combined_frame = np.hstack((frame, panel))
            # Calculate RESET button position.
            button_x = combined_frame.shape[1] - 125
            button_y = 15
            button_w = 105
            button_h = 40
            # Update the coordinates used by the mouse callback.
            button_bounds.update({
                "x": button_x,
                "y": button_y,
                "w": button_w,
                "h": button_h
            })
            # Draw RESET button.
            cv2.rectangle(
                combined_frame,
                (button_x, button_y),
                (button_x + button_w, button_y + button_h),
                (0, 0, 200),
                -1
            )
            cv2.putText(
                combined_frame,
                "RESET",
                (button_x + 15, button_y + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
            # Show the final frame.
            cv2.imshow(WINDOW_NAME, combined_frame)
            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()