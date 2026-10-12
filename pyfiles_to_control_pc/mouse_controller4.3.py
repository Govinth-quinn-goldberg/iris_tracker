
import asyncio
import time

import pyautogui
from bleak import BleakClient, BleakScanner


# ============================================================
# BLUETOOTH SETTINGS
# ============================================================

DEVICE_NAME = "GestureAssist-BLE"

SERVICE_UUID = "7f510001-1b15-4d9c-8f8a-7e9d4c2a1000"

CHARACTERISTIC_UUID = "7f510002-1b15-4d9c-8f8a-7e9d4c2a1000"


# ============================================================
# SENSOR CALIBRATION
# ============================================================

CENTER_X = -2500
CENTER_Y = 550

DEADZONE = 2500

MAX_SPEED = 18
SMOOTHING = 0.15
CURVE = 1.5


# ============================================================
# GESTURE SETTINGS
# ============================================================

GESTURE_THRESHOLD = 6500
NEUTRAL_THRESHOLD = 3500

GESTURE_HOLD_TIME = 0.15
NEUTRAL_HOLD_TIME = 0.25


# ============================================================
# SCROLL SETTINGS
# ============================================================

SCROLL_AMOUNT = 3
SCROLL_INTERVAL = 0.06


pyautogui.PAUSE = 0


# ============================================================
# MOUSE MOVEMENT
# ============================================================

def calculate_speed(value):
    if value == 0:
        return 0

    direction = 1 if value > 0 else -1
    normalized = min(abs(value) / 14000, 1.0)

    return direction * (normalized ** CURVE) * MAX_SPEED


# ============================================================
# GESTURE DETECTION
# ============================================================

def detect_gesture(dx, dy):
    if max(abs(dx), abs(dy)) < GESTURE_THRESHOLD:
        return None

    if abs(dx) >= abs(dy):
        return "RIGHT" if dx < 0 else "LEFT"

    return "FORWARD" if dy < 0 else "BACKWARD"


# ============================================================
# SHORTCUTS
# ============================================================

def execute_shortcut(gesture):
    print(f"Gesture detected: {gesture}")

    if gesture == "RIGHT":
        pyautogui.hotkey("win", "tab")
        print("Windows + Tab -> Task View")

    elif gesture == "LEFT":
        pyautogui.press("esc")
        print("Escape pressed")


# ============================================================
# SCROLLING
# ============================================================

def perform_scroll(gesture):
    if gesture == "FORWARD":
        pyautogui.scroll(-SCROLL_AMOUNT)

    elif gesture == "BACKWARD":
        pyautogui.scroll(SCROLL_AMOUNT)


# ============================================================
# MAIN CONTROLLER
# ============================================================

async def main():
    print("=" * 50)
    print("       GESTUREASSIST CONTROLLER")
    print("=" * 50)

    print(f"Searching for {DEVICE_NAME}...")

    # Discover the device dynamically on every launch
    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME,
        timeout=25
    )

    if device is None:
        print("\nESP32 not found.")
        print("Check that it is powered on and advertising.")
        print("Run python scan_ble.py to inspect nearby devices.")
        return

    print(f"Found: {device.name}")
    print(f"Address: {device.address}")
    print("Connecting over BLE...")

    samples = asyncio.Queue(maxsize=1)

    def notification_handler(sender, data):
        try:
            parts = data.decode("utf-8").strip().split(",")

            if len(parts) != 3:
                return

            x, y, button = map(int, parts)

            # Ignore unexpected button values
            if button not in (0, 1):
                return

            sample = (x, y, button)

            # Keep only the newest sensor reading
            if samples.full():
                try:
                    samples.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            samples.put_nowait(sample)

        except (ValueError, UnicodeDecodeError, asyncio.QueueFull):
            pass

    # Controller state
    smooth_x = 0.0
    smooth_y = 0.0

    gesture_mode = False
    gesture_armed = True

    candidate_gesture = None
    candidate_since = None
    neutral_since = None

    last_scroll_time = 0.0

    try:
        async with BleakClient(device, timeout=25) as client:
            print(f"\nBLE connected: {client.is_connected}")

            # Enable sensor notifications
            await client.start_notify(
                CHARACTERISTIC_UUID,
                notification_handler
            )

            print("Sensor notifications enabled.")
            print("\nCONTROLS")
            print("Button released : Mouse mode")
            print("Button held     : Gesture mode")
            print("Tilt right      : Windows + Tab")
            print("Tilt left       : Escape")
            print("Tilt forward    : Scroll down")
            print("Tilt backward   : Scroll up")
            print("\nPress Ctrl+C to stop.\n")

            while client.is_connected:
                try:
                    x, y, button = await asyncio.wait_for(
                        samples.get(),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    if not client.is_connected:
                        break
                    continue

                dx = x - CENTER_X
                dy = y - CENTER_Y
                now = time.monotonic()

                # ============================================
                # GESTURE MODE: BUTTON HELD
                # ============================================

                if button == 1:

                    if not gesture_mode:
                        print("GESTURE MODE")

                        gesture_mode = True
                        gesture_armed = True

                        candidate_gesture = None
                        candidate_since = None
                        neutral_since = None

                    # Do not move mouse during gestures
                    smooth_x = 0.0
                    smooth_y = 0.0

                    is_neutral = (
                        abs(dx) < NEUTRAL_THRESHOLD
                        and abs(dy) < NEUTRAL_THRESHOLD
                    )

                    # Re-arm one-shot shortcuts only after neutral
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

                    gesture = detect_gesture(dx, dy)

                    if gesture is None:
                        candidate_gesture = None
                        candidate_since = None
                        continue

                    # ----------------------------------------
                    # FORWARD / BACKWARD: REPEATED SCROLL
                    # ----------------------------------------

                    if gesture in ("FORWARD", "BACKWARD"):

                        if gesture != candidate_gesture:
                            candidate_gesture = gesture
                            candidate_since = now
                            continue

                        if (
                            candidate_since is not None
                            and now - candidate_since
                            >= GESTURE_HOLD_TIME
                        ):
                            if (
                                now - last_scroll_time
                                >= SCROLL_INTERVAL
                            ):
                                perform_scroll(gesture)
                                last_scroll_time = now

                        continue

                    # ----------------------------------------
                    # LEFT / RIGHT: ONE-SHOT SHORTCUT
                    # ----------------------------------------

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

                        # Prevent repeats until neutral
                        gesture_armed = False

                        candidate_gesture = None
                        candidate_since = None
                        neutral_since = None

                    continue

                # ============================================
                # MOUSE MODE: BUTTON RELEASED
                # ============================================

                if gesture_mode:
                    print("MOUSE MODE")

                    gesture_mode = False
                    gesture_armed = True

                    candidate_gesture = None
                    candidate_since = None
                    neutral_since = None

                    smooth_x = 0.0
                    smooth_y = 0.0
                    last_scroll_time = 0.0

                if abs(dx) < DEADZONE:
                    dx = 0

                if abs(dy) < DEADZONE:
                    dy = 0

                target_x = calculate_speed(dx)
                target_y = calculate_speed(dy)

                smooth_x = (
                    smooth_x * (1 - SMOOTHING)
                    + target_x * SMOOTHING
                )

                smooth_y = (
                    smooth_y * (1 - SMOOTHING)
                    + target_y * SMOOTHING
                )

                # Preserve your existing cursor direction
                pyautogui.moveRel(
                    int(-smooth_x),
                    int(-smooth_y)
                )

    except asyncio.CancelledError:
        raise

    except Exception as error:
        print("\nBLE/controller error:", repr(error))

    finally:
        print("\nGestureAssist stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nStopping GestureAssist...")
