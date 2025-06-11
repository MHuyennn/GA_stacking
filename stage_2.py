"""
Stage 2: Prepare test set predictions for DL models

This stage will cache the predictions of full-train DL models in `pretrained_dir`
on the test set (data.x_test) into `cache_dir/cache_predictions.npy`.
"""

import os
from __init__ import MODELS
from process_data import ProcessData
from utils import predict_pretrained
import numpy as np

def run_stage_2(data_dir, data_path, pretrained_dir, cache_dir):
    data = ProcessData(data_path)
    if not os.path.exists(data_path):
        data.create_npz_dataset(data_dir, data_path)
    data.load_data()
    data.preprocess_data()
    data.check_lengths()

    # Create CACHE_PREDICTIONS table
    CACHE_PREDICTIONS = np.zeros((data.x_test.shape[0], len(MODELS)))

    # Predict each DL model on test set and save the prediction on its corresponding column
    for i, model_name in enumerate(MODELS):
        y_pred = predict_pretrained(data, model_name, on_test_set=True, pretrained_dir=pretrained_dir)
        CACHE_PREDICTIONS[:, i] = np.argmax(y_pred, axis=1)

    # Save x_predict_test
    save_predict_path = os.path.join(cache_dir, 'cache_predictions.npy')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Saving x_predict_test to {save_predict_path} ...")
    np.save(save_predict_path, CACHE_PREDICTIONS)
    print("Finished!")

if __name__ == "__main__":
    run_stage_2(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        cache_dir='/kaggle/working/cache'
    )
