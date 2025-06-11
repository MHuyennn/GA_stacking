"""
Stage 1: Prepare meta predictions for DL models

This stage generates meta-level predictions by performing 5-fold cross-validation for each DL model.
It also saves the final trained DL models in .keras format with unique names.

Steps:
1. For each DL model in MODELS:
    - Train on 4 folds and predict on the remaining fold.
    - Collect out-of-fold predictions for all training samples.
2. Compile all models' predictions into `x_train_meta`, where each column corresponds to a DL model.
3. Save the meta-feature matrix to `meta_dir/x_train_meta.npy`.
4. Save the final trained DL models to `pretrained_dir` with unique names.

Summary:
Each column `i` of `x_train_meta.npy` contains the 5-fold cross-validated predictions from `MODELS[i]` on the training dataset.
Final DL models are saved as {model_name}_final_{timestamp}.keras.
"""

import os
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from __init__ import MODELS, MAPPING
from process_data import ProcessData
from models import MODEL_FACTORY
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def run_stage_1(data_dir, data_path, pretrained_dir, meta_dir):
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

    # Thiết lập cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Khởi tạo ma trận meta
    x_train_meta = np.zeros((data.x_train.shape[0], len(MODELS)))
    print(f"x_train_meta shape: {x_train_meta.shape}")
    print()

    # Lặp qua từng mô hình DL
    for i, model_name in enumerate(MODELS):
        print(f"Processing model: {model_name}")
        model = MODEL_FACTORY[model_name](img_size=img_size)
        checkpoint = ModelCheckpoint(
            os.path.join(pretrained_dir, f"best_{MAPPING[model_name]}.keras"),
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
        # Lưu mô hình cuối cùng với tên duy nhất
        timestamp = int(time.time())
        final_save_path = os.path.join(pretrained_dir, f"{MAPPING[model_name]}_final_{timestamp}.keras")
        model.save(final_save_path)
        print(f"Final model saved as {final_save_path}")
        
        # Thực hiện cross-validation
        for fold, (train_index, val_index) in enumerate(skf.split(data.x_train, data.y_train)):
            print(f"Running fold: {fold + 1}/{n_splits} ...")
            x_train_fold, x_val_fold = data.x_train[train_index], data.x_train[val_index]
            y_train_fold, y_val_fold = data.y_train[train_index], data.y_train[val_index]

            y_fold_pred = model.predict(x_val_fold, batch_size=32)
            # Gán dự đoán vào ma trận meta
            x_train_meta[val_index, i] = np.argmax(y_fold_pred, axis=1)
        print(f"Model: {model_name} has finished Training and predicting on all folds")
        print()

    # Lưu ma trận meta
    save_meta_path = os.path.join(meta_dir, 'x_train_meta.npy')
    os.makedirs(meta_dir, exist_ok=True)
    print(f"Saving x_train_meta to {save_meta_path} ...")
    np.save(save_meta_path, x_train_meta)
    print("Finished!")

if __name__ == "__main__":
    run_stage_1(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        meta_dir='/kaggle/working/meta'
    )
