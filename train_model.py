import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from pathlib import Path

DATA_DIR = Path("data")
ANGLES_FILE = DATA_DIR / "angles.csv"
LABELS_FILE = DATA_DIR / "labels.csv"
MODEL_FILENAME = "pushup_rf_model.joblib"

def train_and_save_model():
    try:
        angles_df = pd.read_csv(ANGLES_FILE)
        labels_df = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"Ошибка: Не найдены файлы {ANGLES_FILE} или {LABELS_FILE}. Убедитесь, что они находятся в директории 'data'.")
        return

    pushup_labels = ['pushups_up', 'pushups_down']
    
    data_df = angles_df.merge(labels_df, on='pose_id')
    
    pushups_data = data_df[data_df['pose'].isin(pushup_labels)].copy()
    
    if pushups_data.empty:
        print("Ошибка: В файле labels.csv нет меток для 'pushups_up' или 'pushups_down'.")
        return

    X = pushups_data.drop(['pose_id', 'pose'], axis=1)
    y = pushups_data['pose']
    
    y_encoded = y.apply(lambda x: 1 if x == 'pushups_up' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("Начало обучения модели Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Обучение завершено.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность модели на тестовом наборе: {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=['pushups_down', 'pushups_up']))

    joblib.dump(model, MODEL_FILENAME)
    print(f"\nМодель успешно сохранена в файл: {MODEL_FILENAME}")

if __name__ == '__main__':
    train_and_save_model()