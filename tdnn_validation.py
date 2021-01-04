# %% # * Imports *
from Utils import read_file, read_file_3lines, printProgressBar
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Embedding, Concatenate, Activation, Dense, \
    Dropout, SpatialDropout1D, Reshape, GlobalAveragePooling1D, LocallyConnected1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import AUC
from tensorflow.keras.initializers import Constant, RandomUniform
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
from os import path
from scipy.linalg import toeplitz
from keras import backend as K
import time

# root needs to change depending on user
root = 'C:/Users/Vasilis Nikiforidis/Desktop/Thesis/MyCode/'
sys.path.append(root)


# %% # **Hyperparameters:**
# > `emb_size`: skills embedding size
# > `L`: history length
# > `max_epochs`: model training epochs
# > `beta`: learning rate parameter
# > `batch_size`: size of subset of data the model parses at each iteration
# > `dropout_rate`: rate of data to ignore

w2v_emb_size = 100
L = 10
max_epochs = 20
beta = 1e-2
batch_size = 300
dropout_rate = 0.8

data_format = "3lines"  # 4columns, 3lines (4columns has beed deprecated)
# assistment_2009_corrected, assistment_2009_updated, fsaif1tof3, assistment2012_13
data_set = "fsaif1tof3"
data_set_folder = "data/" + data_set + "/" + data_format + "/"
embeddings_folder = "embeddings/" + data_set + "/"
embeddings_version = "fsaif1tof3"  # corrected, updated, fsaif1tof3, 12_13
checkpoints_folder = "checkpoints/"


def read_data(iter=1):
    data_set_path = data_set_folder + data_set
    train_file = data_set_path + "_train" + str(iter) + ".csv"
    valid_file = data_set_path + "_valid" + str(iter) + ".csv"
    # Read embedding data
    emb_file = embeddings_folder + \
        "skill_name_embeddings_"+embeddings_version + \
        "_" + str(w2v_emb_size) + ".csv"
    embeddings = pd.read_csv(emb_file, sep=',', header=None)
    # Add a zero row at the beginning
    embeddings = np.vstack((np.zeros([1, w2v_emb_size]), embeddings))

    """
    Read Train, Validation, Test File
    """
    if data_format == "3lines":
        start_user = 1
        data_train, N_train = read_file_3lines(train_file, start_user)
        start_user += N_train
        data_valid, N_valid = read_file_3lines(valid_file, start_user)
    else:
        data_train, N_train = read_file(train_file)
        data_valid, N_valid = read_file(valid_file)

    return data_train, data_valid, embeddings

# %%
# ### Generate `x_train`, `x_test`, `t_train`, `t_test`
# Every student `stud_id` has a sequence of responses `correct[0], correct[1],..., correct[T-1]` for some
# skill `skill_id`. The length `T` of the sequence depends on the student and the skill.
# Every row of `x_train` or `x_test` contains the `student_id`, the `skill_id` and the response `correct[t]` for
# some time `t`. In addition to that it also includes the history of length `L` of previous responses `correct[t-1],..., correct[t-L]`.
# These responses *must* correspond to the same student and the same skill as time `t`. If history is shorter than `L` then the
# missing entries are filled with `0`.


def gen_inputs_targets(data, user_ids, N, prefix):
    printProgressBar(0, N, prefix=prefix, suffix='Complete', length=50)

    x = None
    t = None
    start = True
    for i, student_id in enumerate(user_ids):
        # Make an array with all the data for this student
        student_data = data[data[:, 0] == student_id]
        skill_hist = toeplitz(student_data[:, 1], np.zeros([1, L]))
        responses_hist = toeplitz(student_data[:, 2], np.zeros([1, L]))
        student_data = np.hstack((skill_hist,
                                  np.fliplr(responses_hist)
                                  ))

        if start:
            start = False
            x = student_data[1:, 0:2*L-1]
            t = student_data[1:, 2*L-1].reshape([-1, 1])
        else:
            x = np.vstack((x, student_data[1:, 0:2*L-1]))
            t = np.vstack((t, student_data[1:, 2*L-1].reshape([-1, 1])))
        printProgressBar(i+1, N, prefix=prefix, suffix='Complete', length=50)
    return x, t

# %% # * Time Delay Neural Network model *
# `L` = time lag


def model_tdnn_w2v(dropout_rate=None, embeddings=np.zeros([1, w2v_emb_size])):
    # Inputs
    query_input = Input(shape=[L], dtype=tf.int32)
    history_input = Input(shape=[L-1], dtype=tf.int32)

    initial_q_emb = Constant(embeddings/(L*w2v_emb_size))
    query_embeddings = Embedding(embeddings.shape[0], w2v_emb_size,
                                 embeddings_initializer=initial_q_emb, mask_zero=True)(query_input)

    initial_h_emb = RandomUniform(
        minval=-1/(w2v_emb_size*L), maxval=1/(w2v_emb_size*L))
    history_embeddings = Embedding(2, w2v_emb_size,
                                   embeddings_initializer=initial_h_emb)(history_input)

    dropout_layer = SpatialDropout1D(dropout_rate)
    query = dropout_layer(query_embeddings)
    history = dropout_layer(history_embeddings)

    query = LocallyConnected1D(
        filters=100, kernel_size=4, padding="valid", activation="relu", implementation=2)(query)
    history = LocallyConnected1D(
        filters=100, kernel_size=4, padding="valid", activation="relu", implementation=2)(history)

    pooling_layer = GlobalAveragePooling1D()
    query_encoding = pooling_layer(query)
    history_encoding = pooling_layer(history)

    x = Concatenate(axis=1)([query_encoding, history_encoding])

    x = Activation(activation="relu")(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(150, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[query_input, history_input], outputs=x)
    return model

# %% # * Callbacks *


model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoints_folder,
    monitor='auc',
    mode='max',
    save_best_only=True,
    save_weights_only=True
)

# %%
# * Model Validation *

validation_results = []
for i in range(5):
    iter = str(i + 1)
    data_train, data_valid, embeddings = read_data(iter)
    train_user_ids = np.unique(data_train[:, 0])
    N_train_users = len(train_user_ids)
    valid_user_ids = np.unique(data_valid[:, 0])
    N_valid_users = len(valid_user_ids)
    x_train, t_train = gen_inputs_targets(data_train,
                                          train_user_ids, N_train_users, 'Train set:')
    x_valid, t_valid = gen_inputs_targets(data_valid,
                                          valid_user_ids, N_valid_users, 'Validation set:')

    # * Train the model *
    acc_valid_base = np.sum(t_valid == 1)/t_valid.shape[0]
    print('Baseline valid accuracy = {}'.format(acc_valid_base))
    print("==================================================")

    model = model_tdnn_w2v(dropout_rate=dropout_rate, embeddings=embeddings)
    model.summary()
    plot_model(model, show_shapes=True, expand_nested=True)

    model.compile(optimizer=Adamax(learning_rate=beta),
                  loss=MeanSquaredError(),
                  metrics=['accuracy', AUC()])

    start = time.time()
    history = model.fit(
        [x_train[:, :L].astype(int), x_train[:, L:].astype(int)],
        t_train,
        validation_data=(
            [x_valid[:, :L].astype(int), x_valid[:, L:].astype(int)],
            t_valid
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[ReduceLROnPlateau(monitor="auc"), model_checkpoint_callback]
    )
    end = time.time()
    print("Model trained in: {} seconds".format(end - start))
    auc_scores = []
    if i == 0:
        auc_scores = history.history.get("val_auc")
    else:
        auc_scores = history.history.get("val_auc_" + str(i))

    validation_results.append(auc_scores[len(auc_scores) - 1])

print("Max Validation Results: {}".format(validation_results))
print("Final Average Validation Score: {}".format(
    np.average(validation_results)))
