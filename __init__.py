# __init__.py (cập nhật)
import gc
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from keras import Sequential
from keras.applications import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, callbacks, losses, models, optimizers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adamax

# Model names (chỉ sử dụng DL models)
DL_MODELS = ['VGG16', 'Xception', 'ResNet50', 'MobileNet']
MODELS = DL_MODELS  # Chỉ chạy DL models

# Pretrained file name mapping (cập nhật cho DL)
MAPPING = {
    'VGG16': 'VGG16',
    'Xception': 'Xception',
    'ResNet50': 'ResNet50',
    'MobileNet': 'MobileNet'
}

# Paths for training
SAVE_DIR = Path('data')
META_DIR = Path('meta')
PRETRAINED_DIR = Path('pretrained')
CACHE_DIR = Path('cache')

# Create directories if they don't exist
for directory in [SAVE_DIR, META_DIR, PRETRAINED_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
