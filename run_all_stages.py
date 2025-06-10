# run_all_stages.py (cập nhật)
import sys
import os
try:
    from __init__ import MODELS
    print("Successfully imported MODELS from __init__.py")
except ImportError as e:
    print(f"Error importing MODELS: {e}")
    raise

try:
    from stage_1 import run_stage_1
    from stage_2 import run_stage_2
    from stage_3 import run_stage_3
    print("Successfully imported all stage functions")
except ImportError as e:
    print(f"Error importing stage functions: {e}")
    raise

def run_all_stages(data_dir, data_path, pretrained_dir, cache_dir, meta_dir, models_to_run=None):
    # Nếu models_to_run không được cung cấp, chạy tất cả các mô hình trong MODELS (chỉ DL)
    if models_to_run is None:
        models_to_run = MODELS
    # Nếu models_to_run là chuỗi, chuyển thành danh sách đơn
    elif isinstance(models_to_run, str):
        models_to_run = [models_to_run]
    # Kiểm tra xem models_to_run có hợp lệ không
    invalid_models = [m for m in models_to_run if m not in MODELS]
    if invalid_models:
        print(f"Warning: The following models are not in MODELS: {invalid_models}")
        models_to_run = [m for m in models_to_run if m in MODELS]
    print(f"Starting automation for models: {models_to_run}")
    
    for model_name in models_to_run:
        print(f"\nRunning all stages for model: {model_name}")
        
        # Stage 1: Train and cross-validate
        print(f"Executing Stage 1 for {model_name}...")
        run_stage_1(
            data_dir=data_dir,
            data_path=data_path,
            pretrained_dir=pretrained_dir,
            meta_dir=meta_dir
        )
        
        # Stage 2: Generate test predictions
        print(f"Executing Stage 2 for {model_name}...")
        run_stage_2(
            data_dir=data_dir,
            data_path=data_path,
            pretrained_dir=pretrained_dir,
            cache_dir=cache_dir
        )
        
        # Stage 3: Genetic algorithm
        print(f"Executing Stage 3 for {model_name}...")
        run_stage_3(
            data_dir=data_dir,
            data_path=data_path,
            pretrained_dir=pretrained_dir,
            cache_dir=cache_dir,
            meta_dir=meta_dir
        )
        print(f"Completed all stages for {model_name}")

if __name__ == "__main__":
    sys.path.append('/kaggle/working/GA_stacking')
    print("Current sys.path:", sys.path)
    run_all_stages(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/small_test_data.npz',
        pretrained_dir='/kaggle/input/pretrained',  # Sử dụng pretrained có sẵn
        cache_dir='/kaggle/working/cache',
        meta_dir='/kaggle/working/meta'
    )
