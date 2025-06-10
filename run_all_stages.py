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

def run_all_stages(data_dir, data_path, pretrained_dir, cache_dir, meta_dir):
    print(f"Starting automation for models: {MODELS}")
    for model_name in MODELS:
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
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        cache_dir='/kaggle/working/cache',
        meta_dir='/kaggle/working/meta'
    )
