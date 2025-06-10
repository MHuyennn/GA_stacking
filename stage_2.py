"""
Stage 2: Generate test predictions

This stage generates test set predictions using the pretrained models from Stage 1.

Steps:
1. For each model in MODELS:
    - Load the pretrained model.
    - Predict on the test set.
    - Save the predictions in the corresponding column of CACHE_PREDICTIONS.
2. Save the prediction matrix to `cache_dir/cache_predictions.npy`.

Summary:
Each column `i` of `cache_predictions.npy` contains the test set predictions from `MODELS[i]` (see __init__.py).
"""

import os
import numpy as np
from __init__ import MODELS, DL_MODELS
from process_data import ProcessData
from utils import predict_pretrained

def run_stage_2(data_dir, data_path, pretrained_dir, cache_dir):
    # Khởi tạo và xử lý dữ liệu
    data = ProcessData(data_path)
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data.create_npz_dataset(data_dir, data_path)
    data.load_data()
    data.preprocess_data()
    data.check_lengths()

    # In hình dạng dữ liệu
    print(f"data.x_train.shape: {data.x_train.shape}, data.y_train.shape: {data.y_train.shape}")
    print(f"data.x_val.shape: {data.x_val.shape}, data.y_val.shape: {data.y_val.shape}")
    print(f"data.x_test.shape: {data.x_test.shape}, data.y_test.shape: {data.y_test.shape}")
    print()

    # Khởi tạo ma trận dự đoán
    CACHE_PREDICTIONS = np.zeros((data.x_test.shape[0], len(MODELS)))
    print(f"CACHE_PREDICTIONS shape: {CACHE_PREDICTIONS.shape}")
    print()

    # Dự đoán trên tập test cho mỗi mô hình
    for i, model_name in enumerate(MODELS):
        print(f"Predicting with model: {model_name}")
        y_pred = predict_pretrained(data, model_name, img_size=299, on_test_set=True, pretrained_dir=pretrained_dir)
        CACHE_PREDICTIONS[:, i] = np.argmax(y_pred, axis=1) if model_name in DL_MODELS else y_pred
        print()

    # Lưu ma trận dự đoán
    save_cache_path = os.path.join(cache_dir, 'cache_predictions.npy')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Saving cache_predictions to {save_cache_path} ...")
    np.save(save_cache_path, CACHE_PREDICTIONS)
    print("Finished!")

if __name__ == "__main__":
    run_stage_2(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        cache_dir='/kaggle/working/cache'
    )
