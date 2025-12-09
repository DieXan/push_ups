import streamlit as st
import cv2
import time
from pose_detector import poseDetector 
import numpy as np
import requests

API_URL = "https://push-ups-7reb.onrender.com/predict_pose" 
CONFIDENCE_THRESHOLD = 0.75 
API_INTERVAL = 0.15 
MIN_DOWN_FRAMES = 3 

KNEE_STRAIGHTNESS_MIN = 170.0 
KNEE_STRAIGHTNESS_MAX = 185.0 

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

if 'detector' not in st.session_state:
    st.session_state.detector = poseDetector()

initial_state = {
    "count": 0,
    "status": "UP",
    "last_pose": "pushups_up",
    "confidence": 0.0,
    "feedback": "Ожидание позы...",
    "in_frame": False,
    "is_form_correct": False,
    "down_state_frames": 0,
    "is_running": False
}

for key, value in initial_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

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

def reset_state():
    st.session_state.count = 0
    st.session_state.last_pose = "pushups_up"
    st.session_state.feedback = "Ожидание..."
    st.session_state.down_state_frames = 0
    st.session_state.status = "UP"
    st.session_state.confidence = 0.0
    st.session_state.in_frame = False
    st.session_state.is_form_correct = False

def start_stop_handler():
    if st.session_state.is_running:
        st.session_state.is_running = False
    else:
        reset_state()
        st.session_state.is_running = True

def run_detector(camera_id, video_placeholder, count_placeholder, status_placeholder, feedback_placeholder, confidence_placeholder, form_placeholder):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        video_placeholder.error(f"Камера ID {camera_id} недоступна. Попробуйте другой ID.")
        st.session_state.is_running = False
        return

    detector = st.session_state.detector
    last_api_request_time = time.time()
    
    while st.session_state.is_running:
        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        img = cv2.flip(img, 1)
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=False)
        
        current_time = time.time()

        if lmList and check_visibility(lmList):
            st.session_state.in_frame = True
            
            if current_time - last_api_request_time > API_INTERVAL:
                angles_to_send = get_angles_from_lmList(lmList, detector)
                
                r_knee_angle = angles_to_send[3]
                l_knee_angle = angles_to_send[4]
                
                is_straight_r_knee = KNEE_STRAIGHTNESS_MIN < r_knee_angle < KNEE_STRAIGHTNESS_MAX
                is_straight_l_knee = KNEE_STRAIGHTNESS_MIN < l_knee_angle < KNEE_STRAIGHTNESS_MAX

                if is_straight_r_knee and is_straight_l_knee:
                    st.session_state.is_form_correct = True
                    try:
                        response = requests.post(API_URL, json={"angles": angles_to_send}, timeout=0.1)
                        response.raise_for_status() 
                        last_api_request_time = current_time
                        
                        prediction_data = response.json()
                        current_pose = prediction_data.get("label", "unknown")
                        confidence = prediction_data.get("confidence", 0.0)
                        st.session_state.confidence = confidence
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            if current_pose == "pushups_down":
                                st.session_state.down_state_frames += 1
                                
                                if st.session_state.down_state_frames >= MIN_DOWN_FRAMES and st.session_state.last_pose == "pushups_up":
                                    st.session_state.status = "ВНИЗ"
                                    st.session_state.last_pose = "pushups_down"
                                    st.session_state.feedback = "ВНИЗ: ТАК ДЕРЖАТЬ!"
                                else:
                                    st.session_state.status = "ВНИЗ (УДЕРЖАНИЕ)"
                                    st.session_state.feedback = f"Удерживайте позу ({st.session_state.down_state_frames}/{MIN_DOWN_FRAMES})"
                            
                            elif current_pose == "pushups_up":
                                st.session_state.down_state_frames = 0 
                                
                                if st.session_state.last_pose == "pushups_down":
                                    st.session_state.count += 1
                                    st.session_state.status = "ВВЕРХ"
                                    st.session_state.last_pose = "pushups_up"
                                    st.session_state.feedback = "ПОВТОР +1! ОТЛИЧНО!"
                                else:
                                    st.session_state.status = "ВВЕРХ (СТАРТ)"
                                    st.session_state.feedback = "Готовы к отжиманию!"
                            
                            else:
                                st.session_state.down_state_frames = 0 
                                st.session_state.status = current_pose.upper().replace('PUSHUPS_', '')
                                st.session_state.feedback = f"ДЕРЖИТЕ ({int(confidence*100)}%)"
                            
                        else:
                            st.session_state.down_state_frames = 0 
                            st.session_state.status = "НЕУВЕРЕННО"
                            st.session_state.feedback = f"Низкая уверенность ({int(confidence*100)}%)"

                    except requests.exceptions.RequestException:
                        st.session_state.down_state_frames = 0
                        st.session_state.feedback = "ОШИБКА API (УБЕДИТЕСЬ, ЧТО FASTAPI ЗАПУЩЕН)"
                        st.session_state.status = "ERROR"
                else:
                    st.session_state.down_state_frames = 0
                    st.session_state.is_form_correct = False
                    st.session_state.status = "НЕ В ПОЗЕ"
                    st.session_state.feedback = f"Выпрямите ноги! ({int(r_knee_angle)}°/{int(l_knee_angle)}°)"
        
        elif lmList and not check_visibility(lmList):
            st.session_state.down_state_frames = 0
            st.session_state.in_frame = False
            st.session_state.status = "НЕ ВСЕ ТОЧКИ"
            st.session_state.feedback = "Встаньте полностью в кадр"
        
        else:
            st.session_state.down_state_frames = 0
            st.session_state.in_frame = False
            st.session_state.status = "НЕТ ЧЕЛОВЕКА"
            st.session_state.feedback = "Камера не видит человека"
        
        
        video_placeholder.image(img, channels="BGR")
        
        count_placeholder.metric("Повторы", st.session_state.count)
        status_placeholder.markdown(f"**Статус:** {st.session_state.status}")
        feedback_placeholder.markdown(f"**Обратная связь:** {st.session_state.feedback}")
        confidence_placeholder.markdown(f"Уверенность модели: **{st.session_state.confidence:.2f}**")
        form_placeholder.markdown(f"Корректность формы: **{'ДА' if st.session_state.is_form_correct else 'НЕТ'}**")
        
    cap.release()
    st.toast("Тренировка остановлена.")


st.set_page_config(layout="wide")
st.title("🏋️ AI Детектор Отжиманий (Streamlit Web)")

col1, col2 = st.columns([3, 1])

with col2:
    st.header("Настройки и Статус")
    
    camera_id_select = st.selectbox("Выберите ID камеры", [0, 1, 2], index=0)
    
    st.button("Начать / Остановить тренировку", on_click=start_stop_handler, 
              type="primary" if not st.session_state.is_running else "secondary") # <-- ИСПРАВЛЕНИЕ: "danger" заменен на "secondary"
              
    st.subheader("Статистика в реальном времени")
    count_placeholder = st.empty()
    status_placeholder = st.empty()
    feedback_placeholder = st.empty()
    confidence_placeholder = st.empty()
    form_placeholder = st.empty()

    count_placeholder.metric("Повторы", st.session_state.count)
    status_placeholder.markdown(f"**Статус:** {st.session_state.status}")
    feedback_placeholder.markdown(f"**Обратная связь:** {st.session_state.feedback}")
    confidence_placeholder.markdown(f"Уверенность модели: **{st.session_state.confidence:.2f}**")
    form_placeholder.markdown(f"Корректность формы: **{'ДА' if st.session_state.is_form_correct else 'НЕТ'}**")


with col1:
    st.header("Видеопоток с Обнаружением Позы")
    video_placeholder = st.empty()

    if st.session_state.is_running:
        run_detector(camera_id_select, video_placeholder, count_placeholder, status_placeholder, feedback_placeholder, confidence_placeholder, form_placeholder)
    else:
        st.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Тренировка остановлена.\nНажмите кнопку 'Начать'.")