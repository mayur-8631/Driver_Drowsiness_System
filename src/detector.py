import cv2

class FaceEyeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            "cascades/haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            "cascades/haarcascade_eye.xml"
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        results = []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 15)

            results.append({
                "face": (x, y, w, h),
                "eyes": eyes
            })

        return results
