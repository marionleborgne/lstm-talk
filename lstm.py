from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import numpy as np


def augment_data(X, y, duplication_ratio):
    """
    See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings
    -using-recurrent-neural-networks/

    :param X: Each row is a training sequence
    :param y: The target we train and will later predict
    :param duplication_ratio: (float) the percentage of data to duplicate.
    :return X, y: augmented data
    """
    nb_duplicates = duplication_ratio * len(X)

    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, nb_duplicates)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def create_model(sequence_length, layers):
    model = Sequential()
    model.add(LSTM(units=layers['hidden1'],
                   input_shape=(sequence_length - 1, layers['input']),
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=layers['hidden2'], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=layers['hidden3'], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=layers['output']))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model


def create_train_and_test(data, sequence_length, split_index,
                          duplication_ratio):
    """

    :param data: (array)
        Data to convert. The last value is the label.
    :param sequence_length: (int)
        Length of the sequence.
    :param split_index: (int)
        Train / test split index.
    :param duplication_ratio: (float)
        Data duplication percentage for the data augmentation step.
    :return X_train, y_train, X_test, y_test: (4-tuple of arrays)
        Train and test sets.
    """
    nb_records = len(data)
    print "Total number of records:", nb_records

    print "Creating train data..."
    result = []
    for index in range(split_index - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape = (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of train data :", result_mean
    print "Train data shape  :", result.shape

    train = result[:split_index, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = augment_data(X_train, y_train, duplication_ratio)

    print "Creating test data..."
    result = []
    for index in range(split_index, nb_records - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape = (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of test data : ", result_mean
    print "Test data shape  : ", result.shape

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test
