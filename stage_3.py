"""
Stage 3: GA training

(Optional) before GA, we pre-calculate the performances of full-trained models
on validation set for weighted population initialization.
You can skip this step by setting model_performances=None, and GA will use random initialization algorithm instead.

For each chromosome (a binary array of 0s and 1s), we denote:

chosen_indexes = [i for i, value in enumerate(chromosome) if value == 1]

We then select columns from chosen_indexes in x_train_meta to get a subset x_chromosome_meta, then
train a separate meta-learner (let's call it M) for this (x_chromosome_meta, y_train_meta)
dataset.

After this we select columns from chosen_indexes in CACHE_PREDICTIONS
to get x_test_meta and use M to predict on it and get y_test_meta.

The f1_score of (y_test_meta, data.y_test) will be the fitness score
of this chromosome, and the winning chromosome after training GA will be the final test set result.
"""

import os
from __init__ import MODELS, DL_MODELS
from process_data import ProcessData
from chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from utils import predict_pretrained
import numpy as np
from sklearn.metrics import f1_score

def run_stage_3(data_dir, data_path, pretrained_dir, cache_dir, meta_dir):
    data = ProcessData(data_path)
    if not os.path.exists(data_path):
        data.create_npz_dataset(data_dir, data_path)
    data.load_data()
    data.preprocess_data()
    data.check_lengths()

    CACHE_PREDICTIONS = np.load(os.path.join(cache_dir, 'cache_predictions.npy'))
    x_train_meta = np.load(os.path.join(meta_dir, 'x_train_meta.npy'))

    params_tuple = (data, CACHE_PREDICTIONS, x_train_meta)

    # Pre-calculate F1 performance scores on validation set (data.x_val) to use for GA's weighted population initialization
    model_performances = [1] * len(MODELS)
    for i, model_name in enumerate(MODELS):
        y_val_pred = predict_pretrained(data, model_name, on_test_set=False, pretrained_dir=pretrained_dir)
        y_val_pred = np.argmax(y_val_pred, axis=1) if model_name in DL_MODELS else y_val_pred
        model_performances[i] = f1_score(data.y_val, y_val_pred, average='macro')

    # Initialize the Genetic Algorithm
    ga = GeneticAlgorithm(
        params_tuple,
        population_size=5,
        generations=20,
        crossover_rate=0.9,
        mutation_rate=0.1,
        tournament_size=3,
        elitism=True,
        model_performances=model_performances
    )

    print(ga.population)

    # Run the Genetic Algorithm
    ga.run()

if __name__ == "__main__":
    run_stage_3(
        data_dir='/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset',
        data_path='/kaggle/working/data/covid19_radiography_data.npz',
        pretrained_dir='/kaggle/working/pretrained',
        cache_dir='/kaggle/working/cache',
        meta_dir='/kaggle/working/meta'
    )
