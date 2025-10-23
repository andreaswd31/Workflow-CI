import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import dagshub

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split # Kita akan split data latih lagi untuk validasi

# --- 1. Konfigurasi DagsHub ---
DAGSHUB_USERNAME = "andreaswd31"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Andreas-Wirawan-Dananjaya"

# Inisialisasi DagsHub
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

# Set URI pelacakan MLflow ke DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

# --- 2. Fungsi Helper ---

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Data latih dimuat: {df_train.shape}")
    print(f"Data uji dimuat: {df_test.shape}")
    
    return df_train, df_test

def split_data(df):
    """Memisahkan fitur (X) dan target (y)."""
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    return X, y

def plot_confusion_matrix(cm, run_id):
    """Membuat plot confusion matrix dan menyimpannya ke file."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Stroke', 'Stroke'],
                yticklabels=['Tidak Stroke', 'Stroke'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    
    # Buat folder 'artifacts' 
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        
    filename = f"artifacts/confusion_matrix_{run_id}.png"
    plt.savefig(filename)
    plt.close() 
    print(f"Confusion matrix disimpan di: {filename}")
    return filename

# --- 3. Fungsi Utama Pelatihan ---

def train_model():
    """Fungsi utama untuk melatih, tuning, dan logging model."""
    
    # Muat data
    df_train, df_test = load_data('train_preprocessed.csv', 'test_preprocessed.csv')
    
    # Pisahkan X dan y
    X_train, y_train = split_data(df_train)
    X_test, y_test = split_data(df_test)

    # --- Hyperparameter Tuning  ---
    print("Memulai Hyperparameter Tuning dengan RandomizedSearchCV...")
    
    # Tentukan model dasar
    rf = RandomForestClassifier(random_state=42)
    
    # Tentukan rentang parameter untuk dicari
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    

    random_search = RandomizedSearchCV(estimator=rf, 
                                       param_distributions=param_dist, 
                                       n_iter=20, cv=3, verbose=1, 
                                       random_state=42, 
                                       n_jobs=-1, 
                                       scoring='recall')
    
    # Latih pencarian
    random_search.fit(X_train, y_train)
    
    # Dapatkan model dan parameter terbaik
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print("Tuning selesai.")
    print(f"Parameter Terbaik: {best_params}")

    # --- Manual Logging (TARGET "ADVANCED") ---
    print("Memulai Manual Logging ke MLflow (DagsHub)...")
    
    # Mulai MLflow Run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # 1. Log Parameter
        # Log parameter terbaik dari RandomizedSearch
        mlflow.log_params(best_params)
        mlflow.log_param("tuning_iterations", 20)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("scoring_strategy", "recall")

        # 2. Lakukan Prediksi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] # Probabilitas kelas positif

        # 3. Hitung Metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # 4. Log Metrik
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        print(f"Metrik di data Uji: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # 5. Log Model
        mlflow.sklearn.log_model(best_model, "model")
        print("Model berhasil di-log.")

        # 6. Log Artefak 
        # Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_confusion_matrix(cm, run_id)
        mlflow.log_artifact(cm_path, "plots") 
        
        # Artefak 2: requirements.txt
        mlflow.log_artifact("requirements.txt")
        
        # Artefak 3: Data Latih 
        mlflow.log_artifact("train_preprocessed.csv", "dataset")

        print("Artefak tambahan berhasil di-log.")
        print("\n--- EKSPERIMEN SELESAI ---")
        print(f"Lihat hasilnya di DagsHub: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    train_model()