import pytest
from fastapi.testclient import TestClient
from api import app, FEATURE_NAMES
import joblib
import numpy as np

# Загружаем тестовую модель, если она существует
try:
    TEST_MODEL = joblib.load('pushup_rf_model.joblib')
except FileNotFoundError:
    # Создаем фиктивную модель и сохраняем, если реальной нет, 
    # чтобы тесты на загрузку не падали
    from sklearn.ensemble import RandomForestClassifier
    # Фиктивные данные для инициализации
    X_dummy = np.array([[100.0] * len(FEATURE_NAMES), [10.0] * len(FEATURE_NAMES)])
    y_dummy = np.array([1, 0])
    TEST_MODEL = RandomForestClassifier(n_estimators=1, random_state=42).fit(X_dummy, y_dummy)
    joblib.dump(TEST_MODEL, 'pushup_rf_model.joblib')
    print("ВНИМАНИЕ: Тестовая модель была создана для тестов. Для реального API запустите train_model.py.")

client = TestClient(app)

def test_health_check():
    """Проверяет работоспособность и загрузку модели."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_pose_success():
    """Проверяет успешное предсказание с корректными данными."""
    # Используем фейковые данные, которые дадут предсказуемый результат
    # Пример "Up" позы (большие углы в локтях)
    up_angles = [170.0, 175.0, 10.0, 160.0, 165.0, 170.0, 175.0]
    response = client.post("/predict_pose", json={"angles": up_angles})
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["label"] in ["pushups_up", "pushups_down"]

def test_predict_pose_wrong_number_of_features():
    """Проверяет, что API возвращает 400 при неверном числе углов."""
    # Передаем только 5 углов, а ожидается 7
    wrong_angles = [10.0, 10.0, 10.0, 10.0, 10.0]
    response = client.post("/predict_pose", json={"angles": wrong_angles})
    
    assert response.status_code == 400
    assert "Ожидается 7 углов" in response.json()["detail"]

def test_predict_pose_not_list():
    """Проверяет, что Pydantic ловит неверный формат входных данных."""
    response = client.post("/predict_pose", json={"angles": "not_a_list"})
    
    assert response.status_code == 422 # Ошибка валидации Pydantic