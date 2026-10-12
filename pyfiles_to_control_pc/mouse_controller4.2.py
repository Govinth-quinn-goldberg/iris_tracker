
import serial
import pyautogui
import time

# ============================================================
# SETTINGS
# ============================================================

PORT = "COM8"
BAUD_RATE = 115200

# MPU6050 neutral position
CENTER_X = -2500
CENTER_Y = 550

# Mouse settings
DEADZONE = 2500
MAX_SPEED = 18
SMOOTHING = 0.15
CURVE = 1.5

# Gesture settings
GESTURE_THRESHOLD = 6500
NEUTRAL_THRESHOLD = 3500
GESTURE_HOLD_TIME = 0.15
NEUTRAL_HOLD_TIME = 0.25

# Scrolling settings
SCROLL_AMOUNT = 25
SCROLL_INTERVAL = 0.06

# ============================================================
# SERIAL CONNECTION
# ============================================================

print("Connecting to ESP32...")

esp32 = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)

pyautogui.PAUSE = 0

print("Connected!")
print("Button released = MOUSE MODE")
print("Button held     = GESTURE MODE")
print()
print("Right    -> Windows + Tab (Task View)")
print("Left     -> Ctrl + Z (Undo)")
print("Forward  -> Scroll Down")
print("Backward -> Scroll Up")
print("Press Ctrl+C to stop.")

# ============================================================
# VARIABLES
# ============================================================

smooth_x = 0
smooth_y = 0

gesture_mode = False
gesture_armed = True

candidate_gesture = None
candidate_since = None
neutral_since = None

last_scroll_time = 0

# ============================================================
# MOUSE SPEED CALCULATION
# ============================================================

def calculate_speed(value):
    if value == 0:
        return 0

    direction = 1 if value > 0 else -1
    magnitude = abs(value)

    normalized = min(magnitude / 14000, 1.0)
    speed = (normalized ** CURVE) * MAX_SPEED

    return direction * speed

# ============================================================
# GESTURE RECOGNITION
# ============================================================

def detect_gesture(dx, dy):
    """
    Identify the dominant tilt direction.
    Returns RIGHT, LEFT, FORWARD, BACKWARD, or None.
    """

    if max(abs(dx), abs(dy)) < GESTURE_THRESHOLD:
        return None

    # Select the dominant axis to reduce diagonal errors.
    if abs(dx) >= abs(dy):
        if dx < 0:
            return "RIGHT"
        return "LEFT"

    if dy < 0:
        return "FORWARD"

    return "BACKWARD"


# ============================================================
# SHORTCUT EXECUTION
# ============================================================

def execute_shortcut(gesture):
    print(f"Gesture detected: {gesture}")

    if gesture == "RIGHT":
        pyautogui.hotkey("win", "tab")
        print("Shortcut: Windows + Tab — Task View")

    elif gesture == "LEFT":
        pyautogui.press("esc")
        print("Shortcut: Escape")


# ============================================================
# SCROLLING
# ============================================================

def perform_scroll(gesture):
    """
    Forward tilting scrolls down.
    Backward tilting scrolls up.
    """

    if gesture == "FORWARD":
        pyautogui.scroll(-SCROLL_AMOUNT)

    elif gesture == "BACKWARD":
        pyautogui.scroll(SCROLL_AMOUNT)


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:

        line = esp32.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if not line or line == "READY":
            continue

        # ESP32 format: X,Y,BUTTON
        try:
            x_str, y_str, button_str = line.split(",")

            x = int(x_str)
            y = int(y_str)
            button_pressed = int(button_str) == 1

        except (ValueError, TypeError):
            continue

        # Difference from neutral sensor position
        dx = x - CENTER_X
        dy = y - CENTER_Y

        # ----------------------------------------------------
        # GESTURE MODE: BUTTON HELD
        # ----------------------------------------------------

        if button_pressed:

            if not gesture_mode:
                print("\nGESTURE MODE")
                gesture_mode = True

                gesture_armed = True
                candidate_gesture = None
                candidate_since = None
                neutral_since = None
                last_scroll_time = 0

            # Disable mouse movement during gesture mode.
            smooth_x = 0
            smooth_y = 0

            now = time.monotonic()

            # Check whether the sensor has returned to neutral.
            is_neutral = (
                abs(dx) < NEUTRAL_THRESHOLD
                and abs(dy) < NEUTRAL_THRESHOLD
            )

            # ------------------------------------------------
            # WAIT FOR NEUTRAL BEFORE ANOTHER SHORTCUT
            # ------------------------------------------------

            if not gesture_armed:

                if is_neutral:
                    if neutral_since is None:
                        neutral_since = now

                    elif (
                        now - neutral_since
                        >= NEUTRAL_HOLD_TIME
                    ):
                        gesture_armed = True
                        candidate_gesture = None
                        candidate_since = None
                        neutral_since = None

                        print("Ready for next gesture")

                else:
                    neutral_since = None

                continue

            # ------------------------------------------------
            # DETECT CURRENT GESTURE
            # ------------------------------------------------

            gesture = detect_gesture(dx, dy)

            if gesture is None:
                candidate_gesture = None
                candidate_since = None
                continue

            # ------------------------------------------------
            # FORWARD/BACKWARD: CONTINUOUS SCROLLING
            # ------------------------------------------------

            if gesture in ("FORWARD", "BACKWARD"):

                # Require a brief stable tilt before scrolling.
                if gesture != candidate_gesture:
                    candidate_gesture = gesture
                    candidate_since = now
                    continue

                if (
                    candidate_since is not None
                    and now - candidate_since
                    >= GESTURE_HOLD_TIME
                ):
                    if now - last_scroll_time >= SCROLL_INTERVAL:
                        perform_scroll(gesture)
                        last_scroll_time = now

                # Keep scrolling while tilted.
                # Stop and re-arm after returning to neutral.
                continue

            # ------------------------------------------------
            # LEFT/RIGHT: ONE-SHOT SHORTCUTS
            # ------------------------------------------------

            if gesture != candidate_gesture:
                candidate_gesture = gesture
                candidate_since = now
                continue

            if (
                candidate_since is not None
                and now - candidate_since
                >= GESTURE_HOLD_TIME
            ):
                execute_shortcut(gesture)

                # Prevent repeated shortcuts until neutral.
                gesture_armed = False
                candidate_gesture = None
                candidate_since = None
                neutral_since = None

            continue

        # ----------------------------------------------------
        # MOUSE MODE: BUTTON RELEASED
        # ----------------------------------------------------

        if gesture_mode:
            print("MOUSE MODE")
            gesture_mode = False

            smooth_x = 0
            smooth_y = 0

            gesture_armed = True
            candidate_gesture = None
            candidate_since = None
            neutral_since = None
            last_scroll_time = 0

        # ----------------------------------------------------
        # NORMAL MOUSE MOVEMENT
        # ----------------------------------------------------

        if abs(dx) < DEADZONE:
            dx = 0

        if abs(dy) < DEADZONE:
            dy = 0

        target_x = calculate_speed(dx)
        target_y = calculate_speed(dy)

        # Smooth movement
        smooth_x = (
            smooth_x * (1 - SMOOTHING)
            + target_x * SMOOTHING
        )

        smooth_y = (
            smooth_y * (1 - SMOOTHING)
            + target_y * SMOOTHING
        )

        # Preserve your existing cursor direction behavior.
        pyautogui.moveRel(
            int(-smooth_x),
            int(-smooth_y)
        )

except KeyboardInterrupt:
    print("\nStopping GestureAssist...")

except serial.SerialException as e:
    print(f"\nSerial connection error: {e}")

finally:
    if esp32.is_open:
        esp32.close()

    print("Serial connection closed.")
