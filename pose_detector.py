import cv2
import mediapipe as mp
import numpy as np

class poseDetector():
    """
    Класс для обнаружения позы человека с помощью Mediapipe.
    """
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Инициализация настроек модели
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Счетчики для оценки выравнивания бедер
        self.hip_bad_frames = 0
        self.hip_good_frames = 0
        self.hip_persist_threshold = 4   

        # Загрузка утилит Mediapipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose  

        # Создание объекта позы
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=1,
                                     smooth_landmarks=self.smooth,
                                     enable_segmentation=False,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """Обнаруживает ключевые точки позы."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks and draw:
            h, w, c = img.shape
            landmarks = self.results.pose_landmarks.landmark
            points = {}
            for id, lm in enumerate(landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)
    
            connections = list(self.mpPose.POSE_CONNECTIONS)
            for connection in connections:
                if connection[0] in points and connection[1] in points:
                    cv2.line(img, points[connection[0]], points[connection[1]], (0, 255, 0), 2)
    
            for pt in points.values():
                cv2.circle(img, pt, 5, (145, 0 , 217), cv2.FILLED)
    
        return img
    
    def findPosition(self, img, draw=True):
        """Извлекает список координат ключевых точек."""
        lmList = []
        if self.results.pose_landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def findAngle(self, p1, p2, p3):
        """Рассчитывает угол между тремя точками (p2 - вершина)."""
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        x3, y3 = p3[1], p3[2]
    
        a = np.array([x1 - x2, y1 - y2], dtype=float)
        b = np.array([x3 - x2, y3 - y2], dtype=float)
    
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            return 180.0
    
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
        return angle