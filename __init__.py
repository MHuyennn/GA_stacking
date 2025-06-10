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

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Model names
DL_MODELS = ['VGG16', 'Xception', 'ResNet50', 'MobileNet']
ML_MODELS = ['AdaBoostClassifier', 'LGBM', 'XGB', 'CatBoost', 'LogisticRegression', 'RandomForestClassifier', 'KNN']
MODELS = ML_MODELS + DL_MODELS

# Pretrained file name mapping
MAPPING = {
    'VGG16': 'VGG16',
    'Xception': 'Xception',
    'ResNet50': 'ResNet50',
    'MobileNet': 'MobileNet',
    'AdaBoostClassifier': 'AdaBoost',
    'LGBM': 'LGBM',
    'XGB': 'XGB',
    'CatBoost': 'CatBoost',
    'LogisticRegression': 'Logistic',
    'RandomForestClassifier': 'RF',
    'KNN': 'KNN'
}

# Paths for training
SAVE_DIR = Path('data')
META_DIR = Path('meta')
PRETRAINED_DIR = Path('pretrained')
CACHE_DIR = Path('cache')

# Create directories if they don't exist
for directory in [SAVE_DIR, META_DIR, PRETRAINED_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
