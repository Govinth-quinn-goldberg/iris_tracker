import serial
import pyautogui
import time

# ============================================================
# SETTINGS
# ============================================================

PORT = "COM8"
BAUD_RATE = 115200

# Your MPU6050 neutral values
CENTER_X = -2500
CENTER_Y = 550

# Dead zone
DEADZONE = 2500

# Cursor sensitivity
SENSITIVITY_X = 12
SENSITIVITY_Y = 12

# Smoothing
SMOOTHING = 0.25


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


# ============================================================
# MOUSE SETTINGS
# ============================================================

pyautogui.PAUSE = 0


# ============================================================
# VARIABLES
# ============================================================

smooth_x = 0
smooth_y = 0


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

        # Ignore READY message
        if line == "READY":
            continue

        try:
            x, y = map(int, line.split(","))

        except ValueError:
            continue


        # ----------------------------------------------------
        # Calculate difference from neutral
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
        # Convert sensor movement to mouse movement
        # ----------------------------------------------------

        target_x = dx / 1000 * SENSITIVITY_X
        target_y = dy / 1000 * SENSITIVITY_Y


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
        # Move mouse
        # ----------------------------------------------------

        if abs(smooth_x) > 0.2 or abs(smooth_y) > 0.2:

            pyautogui.moveRel(
                int(-smooth_x),
                int(-smooth_y)
            )


except KeyboardInterrupt:

    print("\nStopping mouse controller...")

finally:

    esp32.close()