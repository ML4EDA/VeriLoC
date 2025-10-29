import os
import csv
import pickle

import numpy as np

def get_relative_verilog_paths_from_csv(csv_path):
    """Update the paths in the CSV file to be absolute paths."""
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    updated_rows = []
    with open(csv_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            updated_path = os.path.abspath(os.path.join(csv_dir, row[0]))
            updated_rows.append(updated_path)
    return updated_rows

def read_file(file_path, encoding='iso-8859-1'):
    """Read the content of a file."""
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()
    
def remove_comments(code, comment_symbol='//'):
    """Remove comments from the code."""
    lines = code.split('\n')
    cleaned_lines = [line.split(comment_symbol)[0] for line in lines]
    return '\n'.join(cleaned_lines)

def remove_multiple_newlines(code):
    """Remove multiple consecutive newlines from the code."""
    lines = code.split('\n')
    cleaned_lines = []
    previous_line_empty = False
    for line in lines:
        if line.strip() == '':
            if not previous_line_empty:
                cleaned_lines.append(line)
            previous_line_empty = True
        else:
            cleaned_lines.append(line)
            previous_line_empty = False
    return '\n'.join(cleaned_lines)

def preprocess_code(code, preprocessors):
    """Apply a list of preprocessors to the code."""
    for preprocessor in preprocessors:
        code = preprocessor(code)
    return code.strip()

def write_csv(file_path, header, rows):
    """Write rows to a CSV file with the given header."""
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(rows)

def get_congestion_timing_data(file_type='congestion'):
    congestion_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', file_type, 'openROAD_low_level_modules_yosys_v1'))

    congestion_txt_files = []
    for root, dirs, files in os.walk(congestion_data_dir):
        for file in files:
            if file.endswith(f"{file_type}.txt"):
                congestion_txt_files.append(os.path.join(root, file))

    embeddings_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'embeddings', 'openROAD_low_level_modules_yosys_v1_embeddings'))
    rtl_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

    print(f"Number of {file_type} reports found: {len(congestion_txt_files)}\n\nRemoving duplicates...")

    file_dict = {}
    filtered_file_list = []
    for full_path in congestion_txt_files:
        splits = os.path.split(full_path)
        parent_dir, file_name = splits[-2], splits[-1]
        key = f"{os.path.basename(parent_dir)}/{file_name}"
        # key = file_name
        if key not in file_dict:
            file_dict[key] = []
        file_dict[key].append(full_path)

    for key, values in file_dict.items():
        for congestion_txt_file in values:
            embeddings_file_path = congestion_txt_file.replace(f'{file_type}.txt', 'embeddings.pkl')
            embeddings_file_path = embeddings_file_path.replace(congestion_data_dir, embeddings_data_dir)
            if os.path.exists(embeddings_file_path):
                filtered_file_list.append(congestion_txt_file)
                break
                
    congestion_txt_files = filtered_file_list

    print(f"Number of unique {file_type} reports: {len(congestion_txt_files)}")

    json_data = {}
    count = 0

    for congestion_txt_file in congestion_txt_files:
        embeddings_file_path = congestion_txt_file.replace(f'{file_type}.txt', 'embeddings.pkl')
        embeddings_file_path = embeddings_file_path.replace(congestion_data_dir, embeddings_data_dir)
        with open(congestion_txt_file, 'r') as file:
            lines = file.readlines()

        if len(lines) == 0:
            count += 1
            # print(f"Skipping {congestion_txt_file}")

        data = []
        for line in lines:
            parts = line.strip().split()
            verilog_file_path = parts[0]
            verilog_file_path = verilog_file_path.replace("/home", rtl_dataset_dir)
            line_number = int(parts[1])
            importance = float(parts[2])
            data.append((verilog_file_path, line_number, importance))

            if verilog_file_path not in json_data:
                json_data[verilog_file_path] = {
                    f"{file_type}_data_txt": congestion_txt_file,
                    "embeddings_file_path": embeddings_file_path,
                    f"{file_type}_data": {}
                }
            json_data[verilog_file_path][f"{file_type}_data"][line_number] = importance

    return json_data

def load_module_embeddings():
    embeddings_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'embeddings', 'hidden_states_modules.npz'))
    file_embedding_data = np.load(embeddings_save_path)

    updated_data = {}
    for key in list(file_embedding_data.keys()):
        new_key = key.replace("/scratch/rh3884/verilog_code_generation", "..")
        # new_key = key.split('/')[-1]
        updated_data[new_key] = file_embedding_data[key]

    file_embedding_data = updated_data
    return file_embedding_data

def get_train_test_files(json_data=None):
    # File paths
    train_files_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'train_files_congestion.pkl'))
    test_files_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'test_files_congestion.pkl'))

    # Load train files if files exist
    if os.path.exists(train_files_path) and os.path.exists(test_files_path):
        with open(train_files_path, 'rb') as f:
            train_files = pickle.load(f)

        with open(test_files_path, 'rb') as f:
            test_files = pickle.load(f)
        return train_files, test_files
    else:
        # If files don't exist, create dummy train/test split from json_data
        if json_data is not None:
            all_files = list(json_data.keys())
            split_idx = int(0.8 * len(all_files))
            train_files = all_files[:split_idx]
            test_files = all_files[split_idx:]
            return train_files, test_files
        else:
            return [], []