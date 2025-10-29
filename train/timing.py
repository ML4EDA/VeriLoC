import os
import torch
import pickle
import joblib
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Autoencoder import Autoencoder, weights_init

def get_windowed_data(embeddings_array, windowed, window_size, row_indices):
    if windowed:
        padded_embeddings_list = []
        for file, (start, end) in row_indices.items():
            for i in range(start, end):
                window = []
                for j in range(-window_size, window_size + 1):
                    if start <= i + j < end:
                        window.append(embeddings_array[i + j])
                    else:
                        window.append(np.zeros_like(embeddings_array[0]))
                padded_embeddings_list.append(np.concatenate(window))
        embeddings_array = np.array(padded_embeddings_list)
    return embeddings_array

def get_embeddings_labels(files, json_data, file_embedding_data, use_context=True):
    embeddings_list = []
    regression_embeddings_list = []
    labels = []
    regression_values = []
    file_row_indices = {}
    regression_row_indices = {}
    current_index = 0
    regression_index = 0
    remove_comments = True
    window = 5

    for file in files:
        embeddings_file_path = json_data[file]['embeddings_file_path']
        verilog_file_name = file
        if use_context:
            if verilog_file_name in file_embedding_data:
                file_embeddings = file_embedding_data[verilog_file_name][0]
            else:
                file_embeddings = np.zeros((5120,))
        if os.path.exists(embeddings_file_path):
            with open(embeddings_file_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = None
        
        if embeddings is not None:
            num_lines = max(list(embeddings.keys()))
            timing_data = json_data[file]['timing_data']
            valid_lines = set()
            if num_lines > 1000:
                for key in timing_data:
                    for i in range(key - window, key + window + 1):
                        valid_lines.add(int(i))
            else:
                valid_lines = set(range(num_lines+1))

            # Record the start index for the current file
            file_start_index = current_index
            reg_start_index = regression_index
            # Iterate through all lines in the embeddings dict
            for line_no, value in embeddings.items():
                line = value['verilog_code_line']

                if remove_comments and (len(line.strip().split("//")[0]) > 0) and (int(line_no+1) in valid_lines):
                    embedding = value['embedding']
                    if use_context:
                        embedding = np.concatenate((embedding, file_embeddings))
                    embeddings_list.append(embedding)
                    
                    # Check if the line number is in timing_data keys
                    if (line_no+1) in timing_data and timing_data[line_no+1] != 0:
                        labels.append(1)  # Mark as anomaly (red)
                        regression_values.append(timing_data[line_no+1])
                        regression_embeddings_list.append(embedding)
                        regression_index += 1
                    else:
                        labels.append(0)  # Mark as non-anomaly (blue)
                    
                    current_index += 1
            file_row_indices[file] = (file_start_index, current_index)
            regression_row_indices[file] = (reg_start_index, regression_index)
                    
    embeddings_array = np.array(embeddings_list)
    regression_embeddings_array = np.array(regression_embeddings_list)
    labels_array = np.array(labels)
    regression_values_array = np.array(regression_values)

    return embeddings_array, labels_array, regression_embeddings_array, regression_values_array, file_row_indices, regression_row_indices

def main():
    json_data = get_congestion_timing_data("timing")
    file_embedding_data = load_module_embeddings()

    train_files, test_files = get_train_test_files(json_data)
    use_context = True

    train_embeddings, train_labels, train_reg_embeddings, train_reg_values, train_row_indices, reg_train_row_indices = get_embeddings_labels(train_files, json_data, file_embedding_data, use_context)
    test_embeddings, test_labels, test_reg_embeddings, test_reg_values, test_row_indices, reg_test_row_indices = get_embeddings_labels(test_files, json_data, file_embedding_data, use_context)

    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'scaler.pkl'))
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)

    test_embeddings = scaler.transform(test_embeddings)
    train_embeddings = scaler.transform(train_embeddings)
    test_reg_embeddings = scaler.transform(test_reg_embeddings)
    train_reg_embeddings = scaler.transform(train_reg_embeddings)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert embeddings to PyTorch tensors and move to GPU
    train_reg_embeddings_tensor = torch.tensor(train_reg_embeddings, dtype=torch.float32).to(device)
    test_reg_embeddings_tensor = torch.tensor(test_reg_embeddings, dtype=torch.float32).to(device)
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

    autoencoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'autoencoder.pth'))
    autoencoder = torch.load(autoencoder_path)
    autoencoder.to(device)
    
    autoencoder.eval()
    with torch.no_grad(): 
        train_embeddings_reduced = autoencoder.encoder(train_embeddings_tensor).cpu().numpy()  # Move to CPU for numpy
        test_embeddings_reduced = autoencoder.encoder(test_embeddings_tensor).cpu().numpy()
        train_reg_embeddings_reduced = autoencoder.encoder(train_reg_embeddings_tensor).cpu().numpy()  # Move to CPU for numpy
        test_reg_embeddings_reduced = autoencoder.encoder(test_reg_embeddings_tensor).cpu().numpy()

    X_train = train_embeddings_reduced
    y_train = train_labels
    
    X_test = test_embeddings_reduced
    y_test = test_labels

    X_reg_train = train_reg_embeddings_reduced
    y_reg_train = train_reg_values
    
    X_reg_test = test_reg_embeddings_reduced
    y_reg_test = test_reg_values

    windowed = True
    window_size = 3

    # Apply windowing to the training and test data
    X_train = get_windowed_data(X_train, windowed, window_size, train_row_indices)
    X_test = get_windowed_data(X_test, windowed, window_size, test_row_indices)

    windowed = False

    X_reg_train = get_windowed_data(X_reg_train, windowed, window_size, reg_train_row_indices)
    X_reg_test = get_windowed_data(X_reg_test, windowed, window_size, reg_test_row_indices)

    # Compute scale_pos_weight (ratio of majority class to minority class)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=30,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate the XGBoost model
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non Anomolous", "Anomolous"]))

    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Set LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,  # Handles imbalance
        'num_leaves': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1
    }

    # Train LightGBM model
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=500)

    # Predict and evaluate
    y_prob = lgb_model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non Anomolous", "Anomolous"]))

    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # Save the XGBoost model
    xgb_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'xgb_model_timing.pkl'))
    joblib.dump(xgb_model, xgb_model_path)

    # Save the LightGBM model
    lgb_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'lgb_model_timing.txt'))
    lgb_model.save_model(lgb_model_path)

    # Train timing regression model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler as YScaler
    
    timing_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    timing_regressor.fit(X_reg_train, y_reg_train)
    
    # Scale the y values for better regression performance
    y_scaler = YScaler()
    y_reg_train_scaled = y_scaler.fit_transform(y_reg_train.reshape(-1, 1)).ravel()
    
    # Retrain with scaled y values
    timing_regressor.fit(X_reg_train, y_reg_train_scaled)
    
    # Save the timing regressor and y scaler
    timing_regressor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'timing_regressor.pkl'))
    joblib.dump(timing_regressor, timing_regressor_path)
    
    timing_y_scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'timing_y_scaler.pkl'))
    joblib.dump(y_scaler, timing_y_scaler_path)

if __name__ == "__main__":
    main()
