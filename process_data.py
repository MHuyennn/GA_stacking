import os
import numpy as np
from __init__ import *

class ProcessData:
    def __init__(self, data_path_train, data_path_val_test):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.data_path_train = data_path_train
        self.data_path_val_test = data_path_val_test

    def load_data(self):
        if not os.path.exists(self.data_path_train) or not os.path.exists(self.data_path_val_test):
            raise FileNotFoundError(f"Data files not found at {self.data_path_train} or {self.data_path_val_test}")
        loaded_train = np.load(self.data_path_train, allow_pickle=True)
        self.x_train = loaded_train['train_images']
        self.y_train = loaded_train['train_labels']
        loaded_val_test = np.load(self.data_path_val_test, allow_pickle=True)
        self.x_val = loaded_val_test['val_images']
        self.y_val = loaded_val_test['val_labels']
        self.x_test = loaded_val_test['test_images']
        self.y_test = loaded_val_test['test_labels']
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
