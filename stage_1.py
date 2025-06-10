"""
Stage 1: Train all DL models

This stage trains all DL models (VGG16, Xception, ResNet50, MobileNet) on the full training dataset in one go.

Steps:
1. Load and preprocess data.
2. Train each DL model on the full training set.
3. Save each trained model with a unique name to `pretrained_dir` in .keras format.

Summary:
Each model is saved with a unique filename (e.g., VGG16_final.keras) for comparison.
"""

import os
import numpy as np
from __init__ import DL_MODELS, MAPPING, NUM_CLASSES
from process_data import ProcessData
from models import MODEL_FACTORY
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def run_stage_1(data_dir, data_path, pretrained_dir):
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

    img_size = 299  # Đảm bảo khớp với target_size trong process_data.py

    # Lặp qua từng mô hình DL
    for model_name in DL_MODELS:
        print(f"Training model: {model_name}")
        model = MODEL_FACTORY[model_name](img_size=img_size)
        checkpoint = ModelCheckpoint(
            os.path.join(pretrained_dir, f"{MAPPING[model_name]}_final.keras"),
            save_best_only=True,
            monitor='val_loss'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        os.makedirs(pretrained_dir, exist_ok=True)
        model.fit(
            data.x_train,
            data.y_train,
            batch_size=32,
            epochs=50,
            validation_data=(data.x_val, data.y_val),
            callbacks=[checkpoint, early_stopping]
        )
        print(f"Model {model_name} training completed and saved as {MAPPING[model_name]}_final.keras")
        print()

    print("All DL models training finished!")

if __name__ == "__main__":
    run_stage_1(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained'
    )
