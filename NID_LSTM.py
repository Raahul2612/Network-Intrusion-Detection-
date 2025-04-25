import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

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

# Train the LSTM model
def train_lstm_model(dataset_path):
    data = pd.read_csv(dataset_path)
   
    # Encoding categorical features
    encoder = {}
    for col in ["protocol_type", "service", "flag"]:
        encoder[col] = LabelEncoder()
        data[col] = encoder[col].fit_transform(data[col])
   
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["label"])
    y_categorical = to_categorical(y)
   
    X = data.drop(columns=["label"])
   
    # Scaling numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # Reshape for LSTM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
   
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
   
    # LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, X_scaled.shape[1])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(y_categorical.shape[1], activation='softmax')
    ])
   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
   
    # Accuracy evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
   
    class_map = {index: label for index, label in enumerate(label_encoder.classes_)}
    return model, encoder, scaler, X.columns.tolist(), class_map, history

# Evaluate model performance across different packet sets
def evaluate_model_performance(history_list, packet_sizes):
    accuracies = [history.history['val_accuracy'][-1] for history in history_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(packet_sizes, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Packets")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Performance for Different Packet Sets")
    plt.grid()
    plt.show()

# Main function
if __name__ == "__main__":
    dataset_path = "C:/Users/Asus/Downloads/lstm_traffic_dataset.csv"
    packet_sizes = [50, 100, 150, 200, 250]
    history_list = []
    
    for packet_count in packet_sizes:
        print(f"Training model with {packet_count} packets...")
        model, encoder, scaler, expected_features, class_map, history = train_lstm_model(dataset_path)
        history_list.append(history)
    
    print("Class Map:", class_map)
    evaluate_model_performance(history_list, packet_sizes)
