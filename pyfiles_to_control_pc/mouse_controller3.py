
import serial
import pyautogui
import time
import math

# ============================================================
# SETTINGS
# ============================================================

PORT = "COM8"
BAUD_RATE = 115200

# MPU6050 neutral position
CENTER_X = -2500
CENTER_Y = 550

# Dead zone
DEADZONE = 2500

# Maximum cursor speed
MAX_SPEED = 18

# Smoothing
SMOOTHING = 0.15

# Response curve
CURVE = 1.5


# ============================================================
# SERIAL CONNECTION
# ============================================================

print("Connecting to ESP32...")

esp32 = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected!")
print("Button released = MOUSE MODE")
print("Button pressed  = GESTURE MODE")
print("Press Ctrl+C to stop.")

pyautogui.PAUSE = 0


# ============================================================
# VARIABLES
# ============================================================

smooth_x = 0
smooth_y = 0

gesture_mode = False


# ============================================================
# FUNCTION: CONVERT SENSOR TO MOUSE SPEED
# ============================================================

def calculate_speed(value):

    if value == 0:
        return 0

    direction = 1 if value > 0 else -1
    magnitude = abs(value)

    normalized = min(magnitude / 14000, 1.0)

    speed = normalized ** CURVE
    speed *= MAX_SPEED

    return direction * speed


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        line = esp32.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line or line == "READY":
            continue

        # ESP32 sends: X,Y,BUTTON
        try:
            x_str, y_str, button_str = line.split(",")

            x = int(x_str)
            y = int(y_str)
            button_pressed = int(button_str) == 1

        except (ValueError, TypeError):
            continue


        # ----------------------------------------------------
        # BUTTON / MODE CONTROL
        # ----------------------------------------------------

        if button_pressed:

            if not gesture_mode:
                print("GESTURE MODE")
                gesture_mode = True

            # Stop mouse movement while button is held
            smooth_x = 0
            smooth_y = 0

            continue

        else:

            if gesture_mode:
                print("MOUSE MODE")
                gesture_mode = False

                # Start mouse mode without carrying old movement
                smooth_x = 0
                smooth_y = 0


        # ----------------------------------------------------
        # DIFFERENCE FROM NEUTRAL
        # ----------------------------------------------------

        dx = x - CENTER_X
        dy = y - CENTER_Y


        # ----------------------------------------------------
        # DEAD ZONE
        # ----------------------------------------------------

        if abs(dx) < DEADZONE:
            dx = 0

        if abs(dy) < DEADZONE:
            dy = 0


        # ----------------------------------------------------
        # CONVERT TO CURSOR SPEED
        # ----------------------------------------------------

        target_x = calculate_speed(dx)
        target_y = calculate_speed(dy)


        # ----------------------------------------------------
        # SMOOTHING
        # ----------------------------------------------------

        smooth_x = (
            smooth_x * (1 - SMOOTHING)
            + target_x * SMOOTHING
        )

        smooth_y = (
            smooth_y * (1 - SMOOTHING)
            + target_y * SMOOTHING
        )


        # ----------------------------------------------------
        # MOVE CURSOR
        # ----------------------------------------------------

        pyautogui.moveRel(
            int(-smooth_x),
            int(-smooth_y)
        )


except KeyboardInterrupt:

    print("\nStopping mouse controller...")


except serial.SerialException as e:

    print(f"\nSerial connection error: {e}")


finally:

    if esp32.is_open:
        esp32.close()

    print("Serial connection closed.")
