from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

try:
    MODEL = joblib.load('C:\\Users\\DieXan\\Desktop\\maga\\python_labs\\push_ups\\pushup_rf_model.joblib')
    
    FEATURE_NAMES = [
        "right_elbow_right_shoulder_right_hip",
        "left_elbow_left_shoulder_left_hip",
        "right_knee_mid_hip_left_knee", 
        "right_hip_right_knee_right_ankle",
        "left_hip_left_knee_left_ankle",
        "right_wrist_right_elbow_right_shoulder",
        "left_wrist_left_elbow_left_shoulder"
    ]
    MODEL_CLASSES = list(MODEL.classes_)
    CLASS_MAP = {0: "pushups_down", 1: "pushups_up"}
    
except FileNotFoundError:
    print("ПРЕДУПРЕЖДЕНИЕ: Файл pushup_rf_model.joblib не найден. API не будет работать.")
    MODEL = None
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    MODEL = None


app = FastAPI(
    title="Pose Classification API",
    description="API для классификации поз (push-up up/down) на основе углов суставов."
)

class PoseAngles(BaseModel):
    angles: list[float]

@app.get("/health", tags=["Monitoring"])
def get_health():
    """Проверка состояния API и загрузки модели."""
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict_pose", tags=["Prediction"])
def predict_pose(data: PoseAngles):
    """
    Классифицирует позу как 'pushups_up' (1) или 'pushups_down' (0) и возвращает уверенность (confidence).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель машинного обучения не загружена.")
    
    if len(data.angles) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"Ожидается {len(FEATURE_NAMES)} углов, получено {len(data.angles)}.")

    try:
        features = np.array(data.angles).reshape(1, -1)
        
        prediction = MODEL.predict(features)[0] # 0 или 1
        
        probabilities = MODEL.predict_proba(features)[0]
        
        pred_index = MODEL_CLASSES.index(prediction)
        confidence = probabilities[pred_index]
        
        pose_label = CLASS_MAP.get(prediction, "unknown")
        
        return {
            "prediction": int(prediction),
            "label": pose_label,
            "confidence": float(f"{confidence:.4f}"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


if __name__ == "__main__":
    print("Запуск FastAPI сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8000)