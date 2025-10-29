import os
import torch
import pickle
import joblib
import sys

import numpy as np
import xgboost as xgb
import lightgbm as lgb

import sys
import os
from utils import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))
from congestion import get_windowed_data
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Autoencoder import Autoencoder, weights_init

if len(sys.argv) != 2:
    raise ValueError("Please provide the file path as an argument.")

file_path = sys.argv[1]

print(f"Inferring on the file: {file_path}")

infer_embeddings_list = []
infer_labels = []
infer_lines = []
infer_weights = []

use_context = True
windowed = True
window_size = 3

json_data = get_congestion_timing_data()
file_embedding_data = load_module_embeddings()

embeddings_file_path = json_data[file_path]['embeddings_file_path']
verilog_file_name = file_path
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
    raise FileNotFoundError(f"Embeddings file not found for {file_path}")

if embeddings is not None:
    num_lines = max(list(embeddings.keys()))
    congestion_data = json_data[file_path]['congestion_data']

    current_index = 0
    # Iterate through all lines in the embeddings dict
    for line_no, value in embeddings.items():
        line = value['verilog_code_line']
        infer_lines.append(line)
        embedding = value['embedding']
        if use_context:
            embedding = np.concatenate((embedding, file_embeddings))
        infer_embeddings_list.append(embedding)
        # Check if the line number is in congestion_data keys
        if (line_no+1) in congestion_data:
            infer_labels.append(1)  # Mark as anomaly (red)
            infer_weights.append(congestion_data[line_no+1])
        else:
            infer_labels.append(0)  # Mark as non-anomaly (blue)
            infer_weights.append(0)
        current_index += 1

infer_row_indices = {file_path: (0, current_index)}
            
infer_embeddings_array = np.array(infer_embeddings_list)
infer_labels_array = np.array(infer_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'scaler.pkl'))
with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

# Scale the infer embeddings
scaled_infer_embeddings = scaler.transform(infer_embeddings_array)

autoencoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'autoencoder.pth'))
autoencoder = torch.load(autoencoder_path)
autoencoder.to(device)

autoencoder.eval()
with torch.no_grad(): 
    infer_embeddings = autoencoder.encoder(scaled_infer_embeddings).cpu().numpy() 

infer_embeddings = get_windowed_data(infer_embeddings, windowed, window_size, infer_row_indices)

lgb_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'lgb_model_congestion.txt'))
lgb_model = lgb.Booster(model_file=lgb_model_path)

y_prob = lgb_model.predict(infer_embeddings)
y_pred = (y_prob > 0.5).astype(int)

# Print the anomalies based on y_prob
print("\n\nDetected Congestion Lines:")
anomaly_threshold = 0.5
for line, prob in zip(infer_lines, y_prob):
    if prob > anomaly_threshold:
        print(f"{line}, Probability: {prob}")

# Print the anomalies based on y_prob
print("\n\nActual Congestion Lines:")
anomaly_threshold = 0.5
for line, prob, weight in zip(infer_lines, infer_labels_array, infer_weights):
    if prob > anomaly_threshold:
        print(f"{line}, Weightage: {weight}")
