import cv2

from src.face_mesh import FaceMeshDetector
from src.utils import (
    DrowsinessMetrics,
    draw_dashboard,
    draw_eye_contours
)
from src.alert import AlertManager


def main():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector = FaceMeshDetector()
    metrics = DrowsinessMetrics()
    alerts = AlertManager()

    print("Press Q to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect(frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                features = detector.extract_features(
                    frame,
                    face_landmarks
                )

                avg_ear = features["avg_ear"]
                mouth_opening = features["mouth_opening"]

                print(f"EAR: {avg_ear:.3f}")

                metrics.update_eye_state(
                    avg_ear,
                    threshold=0.24
                )

                metrics.update_yawn(
                    mouth_opening,
                    threshold=35
                )

                metrics.update_drowsiness_timer(
                    avg_ear,
                    threshold=0.24
                )

                draw_eye_contours(
                    frame,
                    features["left_eye_points"],
                    features["right_eye_points"]
                )

                draw_dashboard(
                    frame,
                    features,
                    metrics
                )

                alert_level = metrics.get_alert_level()

                alerts.update_alert(alert_level)

                if alert_level == "ALARM":
                    alerts.handle_high_alert(frame)

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            alerts.stop_alarm()

        cv2.imshow(
            "Driver Drowsiness Detection System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    alerts.stop_alarm()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
