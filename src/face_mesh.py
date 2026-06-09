import cv2
import mediapipe as mp
import math


class FaceMeshDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE = [
            33, 160, 158, 133,
            153, 144, 145, 163
        ]

        self.RIGHT_EYE = [
            362, 385, 387, 263,
            373, 380, 374, 381
        ]

        self.UPPER_LIP = 13
        self.LOWER_LIP = 14

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        return self.face_mesh.process(rgb)

    def get_landmark_point(
            self,
            landmark,
            w,
            h):

        return (
            int(landmark.x * w),
            int(landmark.y * h)
        )

    def distance(
            self,
            p1,
            p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def calculate_ear(
            self,
            eye_points):

        if len(eye_points) < 6:
            return 0

        p1 = eye_points[0]
        p2 = eye_points[1]
        p3 = eye_points[2]
        p4 = eye_points[3]
        p5 = eye_points[4]
        p6 = eye_points[5]

        A = self.distance(p2, p6)
        B = self.distance(p3, p5)
        C = self.distance(p1, p4)

        if C == 0:
            return 0

        return (A + B) / (2.0 * C)

    def extract_features(
            self,
            frame,
            face_landmarks):

        h, w, _ = frame.shape

        left_eye_points = []
        right_eye_points = []

        for idx in self.LEFT_EYE:

            lm = face_landmarks.landmark[idx]

            left_eye_points.append(
                self.get_landmark_point(
                    lm,
                    w,
                    h
                )
            )

        for idx in self.RIGHT_EYE:

            lm = face_landmarks.landmark[idx]

            right_eye_points.append(
                self.get_landmark_point(
                    lm,
                    w,
                    h
                )
            )

        left_ear = self.calculate_ear(
            left_eye_points
        )

        right_ear = self.calculate_ear(
            right_eye_points
        )

        avg_ear = (
            left_ear +
            right_ear
        ) / 2

        upper_lip = (
            face_landmarks.landmark[
                self.UPPER_LIP
            ]
        )

        lower_lip = (
            face_landmarks.landmark[
                self.LOWER_LIP
            ]
        )

        upper_point = self.get_landmark_point(
            upper_lip,
            w,
            h
        )

        lower_point = self.get_landmark_point(
            lower_lip,
            w,
            h
        )

        mouth_opening = self.distance(
            upper_point,
            lower_point
        )

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "mouth_opening": mouth_opening,
            "left_eye_points": left_eye_points,
            "right_eye_points": right_eye_points
        }
