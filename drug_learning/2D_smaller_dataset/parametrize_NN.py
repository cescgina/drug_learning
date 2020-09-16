#!/usr/bin/env python
# coding: utf-8
import os
import sys
from itertools import product
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split

PATH = '../'
PATH_DATA = "../datasets/SARS2/"
PATH_FEAT = 'features/'


def construct_optimizer(hparams, HP_OPTIMIZER, HP_LR):
    if hparams[HP_OPTIMIZER] == "adam":
        return tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR])
    elif hparams[HP_OPTIMIZER] == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=hparams[HP_LR])
    elif hparams[HP_OPTIMIZER] == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=hparams[HP_LR])


def train_test_model(hparams, train_data, train_labels, test_data, test_labels, HP_OPTIMIZER, HP_LR, HP_DROPOUT, HP_NEURONS, HP_L2, HP_HIDDEN_LAYERS):
    internal_layers = [tf.keras.layers.Dropout(hparams[HP_DROPOUT])]+[tf.keras.layers.Dense(hparams[HP_NEURONS], kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_L2]), activation='relu') for _ in range(hparams[HP_HIDDEN_LAYERS])]
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(train_data.shape[1], activation='relu', input_shape=(train_data.shape[1],))] +
                                       internal_layers+[tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer=construct_optimizer(hparams, HP_OPTIMIZER, HP_LR), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")])
    model.fit(train_data, train_labels, epochs=10, verbose=2)
    _, results = model.evaluate(test_data, test_labels, verbose=0)
    return results


def run(run_dir, hparams, train_data, train_labels, test_data, test_labels, HP_OPTIMIZER, HP_LR, HP_DROPOUT, HP_NEURONS, HP_L2, HP_HIDDEN_LAYERS):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, train_data, train_labels, test_data, test_labels, HP_OPTIMIZER, HP_LR, HP_DROPOUT, HP_NEURONS, HP_L2, HP_HIDDEN_LAYERS)
        tf.summary.scalar("accuracy", accuracy, step=1)

def main(run_id):
    if run_id == "morgan":
        features = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", "morgan.npy"))
        data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_morgan.csv"))
    elif run_id == "rdkit":
        features = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", "RDKIT.npy"))
        data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_rdkit.csv"))
    elif run_id == "maccs":
        features = np.load(os.path.join("..", "2D", "features", "SARS1_SARS2", "maccs.npy"))
        data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_maccs.csv"))
    elif run_id == "mordred":
        features = pd.read_csv(os.path.join(PATH_FEAT, 'SARS1_SARS2_mordred_clean_no_outliers.csv'))
        features = features.drop(['Unnamed: 0', "Molecule_index"], axis=1)
        features = features.astype("float64")
        data = pd.read_csv(os.path.join(PATH_DATA, "dataset_cleaned_SARS1_SARS2_morgan.csv"))
    else:
        raise ValueError("Unrecognized argument")
    logs_path = os.path.join("hyperparameters", run_id)
    threshold_activity = 20
    active = (data["activity_merged"] < threshold_activity).values.astype(int)

    train_data, test_data, train_labels, test_labels = train_test_split(features, active, stratify=active)
    HP_HIDDEN_LAYERS = hp.HParam("hidden_layers", hp.Discrete(list(range(3, 10))))
    HP_NEURONS = hp.HParam("neurons", hp.Discrete([i for i in range(10, 151, 20)]))
    HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.2, 0.5]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))
    HP_L2 = hp.HParam('l2 regularizer', hp.Discrete([.001, .01]))
    HP_LR = hp.HParam("learning_rate", hp.Discrete([0.001, 0.01, 0.1, 1.0, 10.0]))

    os.makedirs(logs_path, exist_ok=True)
    with tf.summary.create_file_writer(logs_path).as_default():
        hp.hparams_config(hparams=[HP_HIDDEN_LAYERS, HP_NEURONS, HP_DROPOUT, HP_OPTIMIZER, HP_L2, HP_LR],
                          metrics=[hp.Metric("accuracy", display_name='Accuracy')])

    session_num = 0
    looping = product(HP_NEURONS.domain.values, HP_HIDDEN_LAYERS.domain.values, HP_DROPOUT.domain.values, HP_OPTIMIZER.domain.values, HP_L2.domain.values, HP_LR.domain.values)
    total_runs = len(HP_NEURONS.domain.values)*len(HP_HIDDEN_LAYERS.domain.values)*len(HP_DROPOUT.domain.values)*len(HP_OPTIMIZER.domain.values)*len(HP_L2.domain.values)*len(HP_LR.domain.values)
    for neurons, hidden_lay, dropout, opt, l2, lr in looping:
        hp_params = {HP_NEURONS: neurons, HP_HIDDEN_LAYERS: hidden_lay, HP_DROPOUT: dropout, HP_OPTIMIZER: opt, HP_L2: l2, HP_LR: lr}
        if session_num % 10 == 0:
            # clear everything every 10 models to avoid oom errors
            tf.keras.backend.clear_session()
        run_name = f"run_{session_num}"
        print(f"---Starting trial: {run_name} of {total_runs}")
        print({h.name: hp_params[h] for h in hp_params})
        run(logs_path + '/' + run_name, hp_params, train_data, train_labels, test_data, test_labels, HP_OPTIMIZER, HP_LR, HP_DROPOUT, HP_NEURONS, HP_L2, HP_HIDDEN_LAYERS)
        session_num += 1

if __name__ == "__main__":
    main(sys.argv[1])
