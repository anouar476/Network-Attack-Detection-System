import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import joblib
import json

# Check if model and metrics already exist
if not os.path.exists('knn_model.pkl'):
    # Télécharger le dataset
    path = kagglehub.dataset_download("hassan06/nslkdd")
    print("Path to dataset files:", path)

    # Définir les chemins des fichiers de données
    train_data_path = path + '/KDDTrain+.txt'
    test_data_path = path + '/KDDTest+.txt'

    # Charger les données
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Définir les colonnes
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
               'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
               'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
               'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
               'dst_host_srv_rerror_rate', 'attack', 'level']

    train_data.columns = columns
    test_data.columns = columns
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    label_encoders = {}
    categorical_columns = ['protocol_type', 'service', 'flag', 'attack']

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        full_data[column] = label_encoders[column].fit_transform(full_data[column])

    X = full_data.drop(['attack', 'level'], axis=1) 
    y = full_data['attack']  # Cible

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Prédictions sur les données de test
    y_pred = knn.predict(X_test)

    # Calculer les métriques
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    }

    # Sauvegarder le modèle et les préprocesseurs
    joblib.dump(knn, 'knn_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Sauvegarder les métriques
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("Model trained and saved successfully!")
    print("Metrics:", metrics)
else:
    print("Model already exists! Loading saved metrics...")
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    print("Loaded metrics:", metrics)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion", fontsize=16)
plt.xlabel("Prédictions", fontsize=12)
plt.ylabel("Vérités", fontsize=12)
plt.tight_layout()
plt.show()