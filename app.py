import cv2
import time
from flask import Flask, render_template, Response, request, jsonify
from pose_detector import poseDetector 
import numpy as np
import requests


API_URL = "http://127.0.0.1:8000/predict_pose" 
CONFIDENCE_THRESHOLD = 0.75 
API_INTERVAL = 0.15 
MIN_DOWN_FRAMES = 3 

KNEE_STRAIGHTNESS_MIN = 170.0 
KNEE_STRAIGHTNESS_MAX = 185.0 

app = Flask(__name__)

global_detector = poseDetector()

app_state = {
    "count": 0,
    "status": "UP",
    "last_pose": "pushups_up",
    "confidence": 0.0,
    "feedback": "Ожидание позы...",
    "in_frame": False,
    "is_form_correct": False,
    "down_state_frames": 0 
}

P_L_SHOULDER, P_L_ELBOW, P_L_WRIST = 11, 13, 15
P_R_SHOULDER, P_R_ELBOW, P_R_WRIST = 12, 14, 16
P_L_HIP, P_L_KNEE, P_L_ANKLE = 23, 25, 27
P_R_HIP, P_R_KNEE, P_R_ANKLE = 24, 26, 28

REQUIRED_LANDMARKS = [
    P_L_SHOULDER, P_L_ELBOW, P_L_WRIST,
    P_R_SHOULDER, P_R_ELBOW, P_R_WRIST,
    P_L_HIP, P_L_KNEE, P_L_ANKLE,
    P_R_HIP, P_R_KNEE, P_R_ANKLE
]

def get_angles_from_lmList(lmList, detector_instance):
    lmDict = {pt[0]: pt for pt in lmList}
    angles = []
    
    angle_triplets = [
        (P_R_ELBOW, P_R_SHOULDER, P_R_HIP), 
        (P_L_ELBOW, P_L_SHOULDER, P_L_HIP), 
        (P_L_KNEE, P_L_HIP, P_R_KNEE), 
        (P_R_HIP, P_R_KNEE, P_R_ANKLE), 
        (P_L_HIP, P_L_KNEE, P_L_ANKLE), 
        (P_R_WRIST, P_R_ELBOW, P_R_SHOULDER), 
        (P_L_WRIST, P_L_ELBOW, P_L_SHOULDER) 
    ]
    
    for pt1_id, pt2_id, pt3_id in angle_triplets:
        if all(k in lmDict for k in [pt1_id, pt2_id, pt3_id]):
            angle = detector_instance.findAngle(
                lmDict[pt1_id], 
                lmDict[pt2_id], 
                lmDict[pt3_id]
            )
            angles.append(angle)
        else:
            angles.append(180.0) 
    return angles

def check_visibility(lmList):
    present_ids = {pt[0] for pt in lmList}
    return all(req_id in present_ids for req_id in REQUIRED_LANDMARKS)

def generate_frames(camera_id):
    global app_state 
    
    # Сброс состояния при новом подключении
    app_state["count"] = 0
    app_state["last_pose"] = "pushups_up"
    app_state["feedback"] = "Ожидание..."
    app_state["down_state_frames"] = 0 # << Сброс счетчика
    
    last_api_request_time = 0
    
    cap = None
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError(f"Камера ID {camera_id} недоступна.")
    except Exception as e:
        print(f"Исключение: {e}")
        dummy_frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + dummy_frame + b'\r\n')
        if cap: cap.release()
        return

    while True:
        success, img = cap.read()
        if not success:
            time.sleep(0.5)
            continue
        
        img = cv2.flip(img, 1)
        img = global_detector.findPose(img, draw=True)
        lmList = global_detector.findPosition(img, draw=False)
        
        current_time = time.time()
        
        if lmList:
            if check_visibility(lmList):
                app_state["in_frame"] = True
                
                if current_time - last_api_request_time > API_INTERVAL:
                    
                    angles_to_send = get_angles_from_lmList(lmList, global_detector)
                    
                    r_knee_angle = angles_to_send[3]
                    l_knee_angle = angles_to_send[4]
                    
                    # Проверка формы
                    is_straight_r_knee = KNEE_STRAIGHTNESS_MIN < r_knee_angle < KNEE_STRAIGHTNESS_MAX
                    is_straight_l_knee = KNEE_STRAIGHTNESS_MIN < l_knee_angle < KNEE_STRAIGHTNESS_MAX

                    if is_straight_r_knee and is_straight_l_knee:
                        app_state["is_form_correct"] = True
                        try:
                            response = requests.post(API_URL, json={"angles": angles_to_send}, timeout=0.1)
                            response.raise_for_status() 
                            last_api_request_time = current_time
                            
                            prediction_data = response.json()
                            current_pose = prediction_data.get("label", "unknown")
                            confidence = prediction_data.get("confidence", 0.0)
                            app_state["confidence"] = confidence
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                
                                # --- ЛОГИКА ПОДСЧЕТА И ДЕБАУНСИНГА ---
                                if current_pose == "pushups_down":
                                    app_state["down_state_frames"] += 1
                                    
                                    if app_state["down_state_frames"] >= MIN_DOWN_FRAMES and app_state["last_pose"] == "pushups_up":
                                        # Подтвержденный переход ВНИЗ
                                        app_state["status"] = "ВНИЗ"
                                        app_state["last_pose"] = "pushups_down"
                                        app_state["feedback"] = "ВНИЗ: ТАК ДЕРЖАТЬ!"
                                    elif app_state["last_pose"] == "pushups_down":
                                        # ВНИЗ, но еще не засчитано или уже засчитано
                                        app_state["status"] = "ВНИЗ (УДЕРЖАНИЕ)"
                                        app_state["feedback"] = f"Удерживайте позу ({app_state['down_state_frames']}/{MIN_DOWN_FRAMES})"
                                    else:
                                        # ВНИЗ, но не из UP (например, на середине подъема)
                                        app_state["status"] = "ВНИЗ (ЖДУ ПОДТВЕРЖДЕНИЯ)"
                                        app_state["feedback"] = f"Удерживайте позу ({app_state['down_state_frames']}/{MIN_DOWN_FRAMES})"
                                
                                elif current_pose == "pushups_up":
                                    # Переход ВВЕРХ
                                    app_state["down_state_frames"] = 0 # Сброс счетчика
                                    
                                    if app_state["last_pose"] == "pushups_down":
                                        # Засчитываем повтор
                                        app_state["count"] += 1
                                        app_state["status"] = "ВВЕРХ"
                                        app_state["last_pose"] = "pushups_up"
                                        app_state["feedback"] = "ПОВТОР +1! ОТЛИЧНО!"
                                    else:
                                        app_state["status"] = "ВВЕРХ (СТАРТ)"
                                        app_state["feedback"] = "Готовы к отжиманию!"
                                
                                else:
                                    # Если модель не уверена, сбрасываем счетчик вниз
                                    app_state["down_state_frames"] = 0 
                                    app_state["status"] = current_pose.upper().replace('PUSHUPS_', '')
                                    app_state["feedback"] = f"ДЕРЖИТЕ ({int(confidence*100)}%)"
                                
                                # --- КОНЕЦ ЛОГИКИ ПОДСЧЕТА ---

                            else:
                                app_state["down_state_frames"] = 0 
                                app_state["status"] = "НЕУВЕРЕННО"
                                app_state["feedback"] = f"Низкая уверенность ({int(confidence*100)}%)"

                        except requests.exceptions.Timeout:
                            pass 
                        except requests.exceptions.RequestException:
                            app_state["down_state_frames"] = 0
                            app_state["feedback"] = "ОШИБКА API"
                            app_state["status"] = "ERROR"
                    else:
                        app_state["down_state_frames"] = 0
                        app_state["is_form_correct"] = False
                        app_state["status"] = "НЕ В ПОЗЕ"
                        app_state["feedback"] = f"Выпрямите ноги! ({int(r_knee_angle)}°)"
            else:
                app_state["down_state_frames"] = 0
                app_state["in_frame"] = False
                app_state["status"] = "НЕ ВСЕ ТОЧКИ"
                app_state["feedback"] = "Встаньте полностью в кадр"
        else:
            app_state["down_state_frames"] = 0
            app_state["in_frame"] = False
            app_state["status"] = "НЕТ ЧЕЛОВЕКА"
            app_state["feedback"] = "Камера не видит человека"
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    if cap:
        cap.release()

@app.route('/')
def index():
    available_cameras = []
    for i in range(5): 
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
            else:
                if i == 0 and not available_cameras: break
                if available_cameras and i > available_cameras[-1] + 1: break
        except Exception:
            break
            
    if not available_cameras:
        available_cameras = [0]

    return render_template('index.html', camera_ids=available_cameras)

@app.route('/status')
def status():
    """Возвращает текущее состояние приложения в формате JSON."""
    return jsonify(app_state)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)