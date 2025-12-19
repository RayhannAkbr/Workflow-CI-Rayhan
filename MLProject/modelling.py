import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv('namadataset_preprocessing/diabetes_clean.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan ke folder lokal agar bisa di-push ke GitHub
    mlflow.set_tracking_uri("file:./mlruns")
    
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=100, max_depth=5)
        rf.fit(X_train, y_train)
        mlflow.log_metric("accuracy", rf.score(X_test, y_test))
        mlflow.sklearn.log_model(rf, "model")

if __name__ == "__main__":
    train()