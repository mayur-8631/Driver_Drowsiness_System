import cv2
import time

class DrowsinessMetrics:
    def __init__(self):
        self.total_frames = 0
        self.closed_frame_count = 0
        self.closed_frames = 0
        self.blink_count = 0
        self.blink_rate = 0.0
        self.perclos = 0.0
        self.frame_history = []
        self.start_time = time.time()
        self.drowsy_start = None
        self.drowsy_seconds = 0.0
        self.yawn_count = 0
        self._yawn_active = False

    def update_eye_state(self, avg_ear, threshold=0.24):
        self.total_frames += 1

        if avg_ear < threshold:
            self.closed_frame_count += 1
            self.closed_frames += 1
        else:
            if 3 <= self.closed_frame_count <= 15:
                self.blink_count += 1
            self.closed_frame_count = 0

        self.calculate_perclos()
        self.calculate_blink_rate()

    def update_yawn(self, mouth_opening, threshold=35):
        if not hasattr(self, "_yawn_active"):
            self._yawn_active = False

        if mouth_opening > threshold:
            if not self._yawn_active:
                self.yawn_count += 1
                self._yawn_active = True
        else:
            self._yawn_active = False

    def calculate_perclos(self):
        is_closed = self.closed_frame_count > 0
        self.frame_history.append(is_closed)

        if len(self.frame_history) > 300:
            self.frame_history.pop(0)

        if len(self.frame_history) == 0:
            self.perclos = 0.0
            return

        closed_frames = sum(self.frame_history)
        self.perclos = (closed_frames / len(self.frame_history)) * 100

    def calculate_blink_rate(self):
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes < 0.01:
            self.blink_rate = 0.0
            return
        self.blink_rate = self.blink_count / elapsed_minutes

    def update_drowsiness_timer(self, avg_ear, threshold=0.24):
        if avg_ear < threshold:
            if self.drowsy_start is None:
                self.drowsy_start = time.time()
            self.drowsy_seconds = time.time() - self.drowsy_start
        else:
            self.drowsy_start = None
            self.drowsy_seconds = 0.0

    def get_alert_level(self):
        if self.drowsy_seconds >= 2.0:
            return "ALARM"
        elif self.drowsy_seconds > 0.0:
            return "WARNING"
        else:
            return "NORMAL"


def draw_eye_contours(frame, left_eye_points, right_eye_points):
    for eye_points in [left_eye_points, right_eye_points]:
        if len(eye_points) < 6:
            continue

        # Draw green contour around the eye (8 points)
        num_pts = len(eye_points)
        for i in range(num_pts):
            cv2.line(
                frame,
                eye_points[i],
                eye_points[(i + 1) % num_pts],
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Draw horizontal axis line in blue (from outer to inner corners)
        cv2.line(frame, eye_points[0], eye_points[3], (255, 0, 0), 1, cv2.LINE_AA)
        
        # Draw vertical axis lines in blue
        cv2.line(frame, eye_points[1], eye_points[5], (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frame, eye_points[2], eye_points[4], (255, 0, 0), 1, cv2.LINE_AA)


def draw_dashboard(frame, features, metrics):
    left = features["left_ear"]
    right = features["right_ear"]
    avg = features["avg_ear"]

    # Draw semi-transparent background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (320, 395), (30, 30, 30), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    y = 40
    x = 25

    # L-Eye Openness and Status
    text_l = f"L-Eye Openness: {left:.2f} "
    (w1, h1), _ = cv2.getTextSize(text_l, font, scale, thickness)
    cv2.putText(frame, text_l, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    l_state = "[CLOSED]" if left < 0.24 else "[OPEN]"
    l_color = (0, 0, 255) if left < 0.24 else (0, 255, 0)
    cv2.putText(frame, l_state, (x + w1, y), font, scale, l_color, thickness, cv2.LINE_AA)
    y += 30

    # R-Eye Openness and Status
    text_r = f"R-Eye Openness: {right:.2f} "
    (w2, h2), _ = cv2.getTextSize(text_r, font, scale, thickness)
    cv2.putText(frame, text_r, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    r_state = "[CLOSED]" if right < 0.24 else "[OPEN]"
    r_color = (0, 0, 255) if right < 0.24 else (0, 255, 0)
    cv2.putText(frame, r_state, (x + w2, y), font, scale, r_color, thickness, cv2.LINE_AA)
    y += 30

    # Avg Openness
    avg_color = (0, 0, 255) if avg < 0.24 else (0, 255, 0)
    cv2.putText(frame, f"Avg Openness:  {avg:.2f}", (x, y), font, scale, avg_color, thickness, cv2.LINE_AA)
    y += 35

    # System Status (Large, bold)
    alert_level = metrics.get_alert_level()
    if alert_level == "ALARM":
        sys_text = "System: ALARM"
        sys_color = (0, 0, 255)
    elif alert_level == "WARNING":
        sys_text = "System: WARNING"
        sys_color = (0, 255, 255)
    else:
        sys_text = "System: OK"
        sys_color = (0, 255, 0)

    cv2.putText(frame, sys_text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, sys_color, 2, cv2.LINE_AA)
    y += 35

    # Blinks, Blink Rate, PERCLOS, Yawns
    metrics_data = [
        f"Blinks: {metrics.blink_count}",
        f"Blink Rate: {metrics.blink_rate:.1f}/min",
        f"PERCLOS: {metrics.perclos:.1f}%",
        f"Yawns: {metrics.yawn_count}"
    ]
    for m_text in metrics_data:
        cv2.putText(frame, m_text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 28

    y += 10
    # Progress Bar
    bar_x1, bar_y1 = x, y
    bar_width = 150
    bar_height = 12
    bar_x2, bar_y2 = bar_x1 + bar_width, bar_y1 + bar_height

    # Draw dark background bar
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)

    # Draw yellow filled progress bar based on 2.0s limit
    fill_ratio = min(metrics.drowsy_seconds / 2.0, 1.0)
    if fill_ratio > 0:
        fill_x2 = bar_x1 + int(fill_ratio * bar_width)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), (0, 255, 255), -1)

    # Time label next to the bar
    time_text = f"{metrics.drowsy_seconds:.1f}s/2.0s"
    cv2.putText(frame, time_text, (bar_x2 + 15, bar_y1 + 10), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
