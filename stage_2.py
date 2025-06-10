"""
Stage 2: Prepare test set predictions

This stage will cache the predictions of full-train models in `pretrained_dir`
(models trained on 100% train dataset) on the test set (data.x_test)
into `cache_dir/cache_predictions.npy` (let's call it CACHE_PREDICTIONS table)
to reduce the time complexity of step 3.
"""

import os  # Thêm import os
import numpy as np
from __init__ import *
from process_data import ProcessData
from utils import predict_pretrained

def run_stage_2(data_dir, data_path, pretrained_dir, cache_dir):
    data = ProcessData(data_path)
    if not os.path.exists(data_path):
        data.create_npz_dataset(data_dir, data_path)
    data.load_data()
    data.preprocess_data()
    data.check_lengths()

    # Create CACHE_PREDICTIONS table
    CACHE_PREDICTIONS = np.zeros((data.x_test.shape[0], len(MODELS)))

    # Predict each model on test set and save the prediction on its corresponding column
    for i, model_name in enumerate(MODELS):
        y_pred = predict_pretrained(data, model_name, img_size=299, on_test_set=True, pretrained_dir=pretrained_dir)
        CACHE_PREDICTIONS[:, i] = np.argmax(y_pred, axis=1) if model_name in DL_MODELS else y_pred

    # Save x_predict_test
    save_predict_path = os.path.join(cache_dir, 'cache_predictions.npy')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Saving x_predict_test to {save_predict_path} ...")
    np.save(save_predict_path, CACHE_PREDICTIONS)
    print("Finished!")

if __name__ == "__main__":
    # Ví dụ mặc định, có thể thay đổi trong Notebook
    run_stage_2(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        cache_dir='/kaggle/working/cache'
    )
