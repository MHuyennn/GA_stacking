import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from __init__ import *

class ProcessData:
    def __init__(self, data_path=None):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.data_path = data_path
        if self.data_path is None:
            raise ValueError("data_path must be provided when initializing ProcessData.")

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}. Please ensure the data is available.")
        loaded_data = np.load(self.data_path, allow_pickle=True)
        self.x_train = loaded_data['train_images']
        self.y_train = loaded_data['train_labels']
        self.x_val = loaded_data['val_images']
        self.y_val = loaded_data['val_labels']
        self.x_test = loaded_data['test_images']
        self.y_test = loaded_data['test_labels']
        print("Data Loaded!")

    def preprocess_data(self):
        def normalize(data):
            return data / 255.0
        self.x_train = normalize(self.x_train)
        self.x_val = normalize(self.x_val)
        self.x_test = normalize(self.x_test)
        print("Data Processed!")

    def check_lengths(self):
        assert len(self.x_train) == len(self.y_train), f"Mismatch in train data length: {len(self.x_train)} images vs {len(self.y_train)} labels"
        assert len(self.x_val) == len(self.y_val), f"Mismatch in validation data length: {len(self.x_val)} images vs {len(self.y_val)} labels"
        assert len(self.x_test) == len(self.y_test), f"Mismatch in test data length: {len(self.x_test)} images vs {len(self.y_test)} labels"
        print("All data lengths match!")

    def create_npz_dataset(self, data_dir, output_path_train, output_path_val_test, categories=['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity'], target_size=(299, 299)):
        x_data, y_data = [], []
        print(f"Scanning data from {data_dir}")
        
        for idx, category in enumerate(categories):
            images_folder = os.path.join(data_dir, category, 'images')
            if not os.path.exists(images_folder):
                print(f"Warning: Folder {images_folder} not found. Skipping...")
                continue
            img_filenames = sorted([f for f in os.listdir(images_folder) if f.lower().endswith('.png')])
            print(f"Processing {category}/images with {len(img_filenames)} files")
            
            for img_filename in img_filenames:
                img_path = os.path.join(images_folder, img_filename)
                try:
                    print(f"Attempting to load {img_path}")
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Failed to load image {img_path}")
                    print(f"Image loaded with shape {img.shape}")
                    img_resized = cv2.resize(img, target_size)
                    if img_resized.shape != (299, 299):
                        raise ValueError(f"Unexpected shape {img_resized.shape} for {img_path}, expected (299, 299)")
                    print(f"Image resized to shape {img_resized.shape}")
                    # Chuyển ảnh xám sang 3 kênh bằng cách lặp lại kênh
                    img_array = np.stack([img_resized] * 3, axis=-1).astype("float32") / 255.0
                    if img_array.shape != (299, 299, 3):
                        raise ValueError(f"Unexpected final shape {img_array.shape} for {img_path}, expected (299, 299, 3)")
                    print(f"Image array created with shape {img_array.shape}")
                    x_data.append(img_array)
                    y_data.append(idx)
                    print(f"Successfully loaded {img_path} with shape {img_array.shape} (appended to x_data, x_data length: {len(x_data)})")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        if not x_data:
            raise ValueError(f"No valid image data found in {data_dir}. Please check the directory or ensure .png files exist. x_data length: {len(x_data)}")
        print(f"Converting x_data to numpy array with shape {np.array(x_data).shape}")
        x_data, y_data = np.array(x_data), np.array(y_data)
        print(f"Total samples collected: {len(x_data)}")
        x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3, stratify=y_data, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        # Lưu tập huấn luyện
        os.makedirs(os.path.dirname(output_path_train), exist_ok=True)
        np.savez(output_path_train, train_images=x_train, train_labels=y_train)
        print(f"Đã tạo và lưu tập huấn luyện vào {output_path_train}")
        
        # Lưu tập xác thực và kiểm tra
        os.makedirs(os.path.dirname(output_path_val_test), exist_ok=True)
        np.savez(output_path_val_test, val_images=x_val, val_labels=y_val, test_images=x_test, test_labels=y_test)
        print(f"Đã tạo và lưu tập xác thực và kiểm tra vào {output_path_val_test}")
