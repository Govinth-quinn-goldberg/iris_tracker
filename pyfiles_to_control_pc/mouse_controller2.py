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
print("Move the controller to control the mouse.")
print("Press Ctrl+C to stop.")

pyautogui.PAUSE = 0


# ============================================================
# VARIABLES
# ============================================================

smooth_x = 0
smooth_y = 0


# ============================================================
# FUNCTION: CONVERT SENSOR TO MOUSE SPEED
# ============================================================

def calculate_speed(value):

    if value == 0:
        return 0

    # Direction
    direction = 1 if value > 0 else -1

    # Magnitude
    magnitude = abs(value)

    # Normalize
    normalized = min(magnitude / 14000, 1.0)

    # Apply response curve
    speed = normalized ** CURVE

    # Convert to mouse speed
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

        if not line:
            continue

        if line == "READY":
            continue

        try:
            x, y = map(int, line.split(","))

        except ValueError:
            continue


        # ----------------------------------------------------
        # Difference from neutral
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
        # Convert to speed
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

    print("\nStopping...")


finally:

    esp32.close()