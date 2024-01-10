import os
import datetime
import seaborn as sns
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv1D, Concatenate, Activation, Multiply, MaxPooling1D, AveragePooling1D, Flatten

from utils import load_data, plot_history_tf, plot_heat_map, plotConfusionMatrix, calculate_metrics

from imblearn.over_sampling import RandomOverSampler


project_path = ""#road
model_path = project_path + "ecg_model.h5"


def buildModel(learning_rate):
    inputs = tf.keras.layers.Input(shape=())#shape
    x = Conv1D(filters=4, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    conv_1 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    conv_2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    conv_3 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    conv_4 = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    conv_5 = Conv1D(filters=16, kernel_size=7, strides=1, padding='same', activation='relu')(conv_1)
    conv_6 = Conv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')(conv_2)
    conv_7 = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(conv_3)
    concatenated = Concatenate()([conv_5, conv_6, conv_7, conv_4])
    concatenated = tf.keras.layers.Dense(64, activation='sigmoid')(concatenated)
    num_heads = 2
    d_model = 16
    d_k = d_model

    def causal_conv1d(filters, kernel_size, strides=1, dilation_rate=1):
        padding = (kernel_size - 1) * dilation_rate
        causal_padding = tf.keras.layers.ZeroPadding1D(padding)(concatenated)
        causal_conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid',
                             dilation_rate=dilation_rate)(causal_padding)
        return causal_conv

    attention_heads = []
    for i in range(num_heads):
        q = causal_conv1d(d_model, 1)
        k = causal_conv1d(d_model, 1)
        v = causal_conv1d(d_model, 1)
        attention_heads.append(tf.keras.layers.Attention(use_scale=True)([q, k, v]))

    attention = tf.keras.layers.Concatenate()(attention_heads)
    attention = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(attention)
    attention = Activation('sigmoid')(attention)
    attention_applied = Multiply()([concatenated, attention])

    x = tf.keras.layers.Flatten()(attention_applied)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def main():

    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)

    if os.path.exists(model_path):
        print('Import the pre-trained model, skip the training process')
        model = tf.keras.models.load_model(filepath=model_path)
    else:

        model = buildModel(learning_rate=LEARNING_RATE)
        model.summary()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        model.save(filepath=model_path)
        plot_history_tf(history)



if __name__ == '__main__':
    main()
