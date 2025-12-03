import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_save_model():
    # 1. Загрузка данных
    try:
        angles_df = pd.read_csv('C:\\Users\\DieXan\\Desktop\\maga\\python_labs\\push_ups\\angles.csv')
        labels_df = pd.read_csv('C:\\Users\\DieXan\\Desktop\\maga\\python_labs\\push_ups\\labels.csv')
    except FileNotFoundError:
        print("Ошибка: Не найдены файлы angles.csv или labels.csv. Убедитесь, что они в той же директории.")
        return

    # 2. Объединение и фильтрация данных для Push-ups
    # Метки для пуш-апов
    pushup_labels = ['pushups_up', 'pushups_down']
    
    # Объединяем признаки (углы) с метками
    data_df = angles_df.merge(labels_df, on='pose_id')
    
    # Фильтруем только строки, относящиеся к пуш-апам
    pushups_data = data_df[data_df['pose'].isin(pushup_labels)].copy()
    
    if pushups_data.empty:
        print("Ошибка: В файле labels.csv нет меток для 'pushups_up' или 'pushups_down'.")
        return

    # 3. Подготовка данных для обучения
    X = pushups_data.drop(['pose_id', 'pose'], axis=1)
    y = pushups_data['pose']
    
    # Преобразование меток: 'pushups_up' -> 1, 'pushups_down' -> 0
    y_encoded = y.apply(lambda x: 1 if x == 'pushups_up' else 0)

    # Разделение на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 4. Обучение модели
    print("Начало обучения модели Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Обучение завершено.")

    # 5. Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность модели на тестовом наборе: {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=['pushups_down', 'pushups_up']))

    # 6. Сохранение модели
    model_filename = 'pushup_rf_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nМодель успешно сохранена в файл: {model_filename}")

if __name__ == '__main__':
    train_and_save_model()