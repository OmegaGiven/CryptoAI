import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error


import csv

def parse_csv_file(csv_fp):
    """
    Takes a csv file specified by the path csv_fp and
    converts it into an array of examples, each of which
    is a dictionary of key-value pairs where keys are
    column names and the values are column attributes.
    """
    examples = []
    with open(csv_fp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        key_names = None
        for row in csv_reader:
            if len(row) == 0:
                continue
            if line_count == 0:
                key_names = row
                for i in range(len(key_names)):
                    ## strip whitespace on both ends.
                    row[i] = row[i].strip()
                    line_count += 1
            else:
                ex = {}
                for i, k in enumerate(key_names):
                    ## strip white spaces on both ends.
                    ex[k] = row[i].strip()
                examples.append(ex)
        return examples, key_names


data, keynames = parse_csv_file("Binance_ETHUSDT_1h.csv")
data.reverse()
# print(data)



def train_test_split(df, test_size=0.2):
    target_col = 'Volume ETH'
    split_row = len(df) - int(test_size * len(df))
    train_data = df
    test_data = df
    volume_list1 = [float(i.get(target_col)) for i in train_data]
    volume_list2 = [float(i.get(target_col)) for i in test_data]
    return volume_list1, volume_list2


train, test = train_test_split(data, test_size=0.2)


target_col = 'Volume ETH'
# volume_list1 = [float(i.get(target_col)) for i in train]
# volume_list2 = [float(i.get(target_col)) for i in test]
# for x in range(len(volume_list1)):
#     volume_list2.insert(0,0)


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('Volume ETH', fontsize=14)
    ax.set_xlabel('hours since 11/22/2020')
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show();

# line_plot(volume_list1, volume_list2, 'training', 'test', title='')

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        window_data += (tmp)
    return window_data

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[window_len:]
    y_test = test_data[window_len:]
    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'

volume_list1, volume_list2, X_train, X_test, y_train, y_test = prepare_data(
    data, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)

# history = model.fit(
#     X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)

preds = test[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)

