import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated network packet
def generate_packet():
    return {
        "duration": random.randint(0, 1000),
        "protocol_type": random.choice(["tcp", "udp", "icmp"]),
        "service": random.choice(["http", "ftp", "ssh", "dns", "smtp"]),
        "flag": random.choice(["SF", "S0", "REJ", "RSTO", "SH"]),
        "src_bytes": random.randint(0, 10000),
        "dst_bytes": random.randint(0, 10000),
        "serror_rate": random.uniform(0, 1),
        "srv_serror_rate": random.uniform(0, 1),
        "rerror_rate": random.uniform(0, 1),
        "srv_rerror_rate": random.uniform(0, 1),
        "same_srv_rate": random.uniform(0, 1),
        "diff_srv_rate": random.uniform(0, 1),
    }

# Train the Random Forest model
def train_rf_model(dataset_path):
    data = pd.read_csv(dataset_path)
   
    # Encoding categorical features
    encoder = {}
    for col in ["protocol_type", "service", "flag"]:
        encoder[col] = LabelEncoder()
        data[col] = encoder[col].fit_transform(data[col])
   
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["label"])
   
    X = data.drop(columns=["label"])
   
    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   
    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
   
    # Accuracy evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
   
    class_map = {index: label for index, label in enumerate(label_encoder.classes_)}
    return model, encoder, scaler, X.columns.tolist(), class_map, accuracy

# Evaluate model performance across different packet sets
def evaluate_model_performance(accuracy_list, packet_sizes):
    plt.figure(figsize=(10, 6))
    plt.plot(packet_sizes, accuracy_list, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Packets")
    plt.ylabel("Validation Accuracy")
    plt.title("Random Forest Model Performance for Different Packet Sets")
    plt.grid()
    plt.show()

# Main function
if __name__ == "__main__":
    dataset_path = "C:/Users/sraah/Downloads/rf_traffic_dataset.csv"
    packet_sizes = [50, 100, 150, 200, 250]
    accuracy_list = []
   
    for packet_count in packet_sizes:
        print(f"Training model with {packet_count} packets...")
        model, encoder, scaler, expected_features, class_map, accuracy = train_rf_model(dataset_path)
        accuracy_list.append(accuracy)
   
    print("Class Map:", class_map)
    evaluate_model_performance(accuracy_list, packet_sizes)
