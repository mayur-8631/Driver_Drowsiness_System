import winsound
import threading
import time
import os
import cv2


class AlertManager:

    def __init__(self):

        self.alarm_active = False
        self.alarm_thread = None

        self.last_log_time = 0

    def _alarm_loop(self):

        while self.alarm_active:
            winsound.Beep(2500, 500)
            time.sleep(0.2)

    def start_alarm(self):

        if self.alarm_active:
            return

        self.alarm_active = True

        self.alarm_thread = threading.Thread(
            target=self._alarm_loop,
            daemon=True
        )

        self.alarm_thread.start()

    def stop_alarm(self):

        self.alarm_active = False

    def update_alert(self, alert_level):

        if alert_level == "ALARM":
            self.start_alarm()
        else:
            self.stop_alarm()

    def save_screenshot(self, frame):

        try:

            folder = "outputs/screenshots"
            os.makedirs(folder, exist_ok=True)

            filename = time.strftime(
                "drowsy_%Y%m%d_%H%M%S.jpg"
            )

            path = os.path.join(folder, filename)

            cv2.imwrite(path, frame)

            return path

        except Exception:
            return None

    def log_event(self, message):

        try:

            folder = "outputs/logs"
            os.makedirs(folder, exist_ok=True)

            log_file = os.path.join(
                folder,
                "drowsiness_log.txt"
            )

            with open(
                log_file,
                "a",
                encoding="utf-8"
            ) as f:

                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                f.write(
                    f"[{timestamp}] {message}\n"
                )

        except Exception:
            pass

    def handle_high_alert(self, frame):

        current_time = time.time()

        if current_time - self.last_log_time < 10:
            return

        self.last_log_time = current_time

        screenshot_path = self.save_screenshot(frame)

        self.log_event(
            f"HIGH DROWSINESS ALERT | Screenshot: {screenshot_path}"
        )
