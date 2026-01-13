#QWOP Vision + Input Prototype
#Screen capture using MSS
import time
import csv
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import mss
import pyautogui

pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort instantly


@dataclass
class TrackingResult:
    center: Optional[Tuple[int, int]]  # (x, y) in ROI coordinates
    area: float
    method: str


def countdown(msg: str, seconds: int = 3):
    print(msg)
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def pick_roi() -> dict:
    """
    Pick ROI by capturing mouse position twice.
    Returns an MSS monitor dict: {"left":..., "top":..., "width":..., "height":...}
    """
    countdown("Move mouse to TOP-LEFT of the game area / capture region.", 3)
    x1, y1 = pyautogui.position()
    print(f"Top-left: ({x1}, {y1})")

    countdown("Move mouse to BOTTOM-RIGHT of the game area / capture region.", 3)
    x2, y2 = pyautogui.position()
    print(f"Bottom-right: ({x2}, {y2})")

    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if width < 50 or height < 50:
        raise ValueError("ROI too small. Re-run and select a larger region.")

    roi = {"left": left, "top": top, "width": width, "height": height}
    print(f"ROI set: {roi}")
    return roi


def track_motion(bgsub, frame_bgr: np.ndarray) -> TrackingResult:
    """
    Foreground/motion-based tracking using background subtraction + largest contour.
    This is a strong “feature extraction + positional tracking” baseline.
    """
    fg = bgsub.apply(frame_bgr)
    fg = cv2.medianBlur(fg, 5)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return TrackingResult(center=None, area=0.0, method="motion")

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 300:  # ignore tiny noise blobs
        return TrackingResult(center=None, area=area, method="motion")

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return TrackingResult(center=(cx, cy), area=area, method="motion")


def track_color_hsv(frame_bgr: np.ndarray,
                    hsv_lower: Tuple[int, int, int],
                    hsv_upper: Tuple[int, int, int]) -> TrackingResult:
    """
    Optional colour segmentation tracking.
    You can tune hsv_lower/hsv_upper to a distinctive colour on the player.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return TrackingResult(center=None, area=0.0, method="color")

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 200:
        return TrackingResult(center=None, area=area, method="color")

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return TrackingResult(center=(cx, cy), area=area, method="color")


class KeyController:
    """
    Simple input simulation harness.
    This is NOT “trained AI”, but demonstrates consistent keyboard interaction feasibility.
    """
    def __init__(self):
        self.pressed = set()

    def down(self, key: str):
        if key not in self.pressed:
            pyautogui.keyDown(key)
            self.pressed.add(key)

    def up(self, key: str):
        if key in self.pressed:
            pyautogui.keyUp(key)
            self.pressed.remove(key)

    def release_all(self):
        for k in list(self.pressed):
            self.up(k)


def toy_policy(prev_center, center, keys: KeyController):
    """
    A minimal “policy” purely to prove the pipeline:
    - If player appears to move right: tap a pair (q+w) briefly
    - If player moves left or stalls: tap (o+p) briefly
    This is just a feasibility demo for input simulation + closed-loop control.
    """
    if prev_center is None or center is None:
        keys.release_all()
        return "no-track"

    dx = center[0] - prev_center[0]

    # Release everything first (keeps it stable + repeatable)
    keys.release_all()

    if dx > 2:
        keys.down("q")
        keys.down("w")
        return "press-qw"
    elif dx < -2:
        keys.down("o")
        keys.down("p")
        return "press-op"
    else:
        # tiny movement = do nothing (or you can alternate)
        return "idle"


def main():
    print("QWOP Vision + Input Prototype (screen capture, feature extraction, positional tracking, input simulation)")
    print("Controls: [S] start/stop input simulation | [C] toggle color tracking overlay | [Q] quit")
    print("Safety: Move mouse to TOP-LEFT corner any time to abort (PyAutoGUI failsafe).")

    roi = pick_roi()

    # Tracking configuration
    use_color_overlay = False

    # These HSV bounds are placeholders.
    # If you want colour segmentation, tweak these using the on-screen mask idea.
    # Skin tone for QWOP character 
    hsv_lower = (20, 80, 80)
    hsv_upper = (35, 255, 255) 


    bgsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    keys = KeyController()

    sim_enabled = False
    prev_center = None

    # Logging
    log_path = "tracking_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "method", "cx", "cy", "area", "sim_enabled", "action"])

        with mss.mss() as sct:
            last_time = time.time()
            fps = 0.0

            while True:
                img = np.array(sct.grab(roi))  # BGRA
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                motion_res = track_motion(bgsub, frame)
                color_res = track_color_hsv(frame, hsv_lower, hsv_upper)

                # Choose primary tracking result:
                # Motion is default; if it fails but color works, fall back to color.
                primary = motion_res if motion_res.center is not None else color_res

                # Overlay
                disp = frame.copy()
                if motion_res.center is not None:
                    cv2.circle(disp, motion_res.center, 6, (0, 255, 0), -1)
                    cv2.putText(disp, f"motion area={motion_res.area:.0f}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if use_color_overlay:
                    # show color center if available
                    if color_res.center is not None:
                        cv2.circle(disp, color_res.center, 6, (255, 0, 0), -1)
                        cv2.putText(disp, f"color area={color_res.area:.0f}", (10, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # FPS
                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
                cv2.putText(disp, f"FPS: {fps:.1f} | sim: {'ON' if sim_enabled else 'OFF'}",
                            (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Input simulation (only when enabled)
                action = ""
                if sim_enabled:
                    action = toy_policy(prev_center, primary.center, keys)
                else:
                    keys.release_all()

                prev_center = primary.center

                # Log primary
                cx, cy = (primary.center if primary.center is not None else (None, None))
                writer.writerow([now, primary.method, cx, cy, f"{primary.area:.2f}", sim_enabled, action])

                cv2.imshow("QWOP Prototype - Vision/Tracking", disp)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("b"), ord("B")):
                    break
                if k in (ord("s"), ord("S")):
                    sim_enabled = not sim_enabled
                    if not sim_enabled:
                        keys.release_all()
                    print(f"Input simulation: {'ENABLED' if sim_enabled else 'DISABLED'}")
                if k in (ord("c"), ord("C")):
                    use_color_overlay = not use_color_overlay
                    print(f"Color overlay: {'ON' if use_color_overlay else 'OFF'}")

    keys.release_all()
    cv2.destroyAllWindows()
    print(f"Done. Tracking log saved to: {log_path}")


if __name__ == "__main__":
    main()
