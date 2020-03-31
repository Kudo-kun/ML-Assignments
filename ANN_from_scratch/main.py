import pandas as pd
import numpy as np
from Layers import dense_layer
from Model import nn_sequential_model
import DataPreprocessor as dp


columns = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "T"]
raw_df = pd.read_csv("housepricedata.txt",
                     sep=",",
                     header=None,
                     names=columns).sample(frac=1)

X_train, Y_train, x_test, y_test = dp.train_test_split(raw_df,
                                                       xfeatures=10,
                                                       split_ratio=0.8,
                                                       normalize=True)

ann = nn_sequential_model()
ann.add_layer(dense_layer(10, activation="linear"), initialization="uniform")
ann.add_layer(dense_layer(5, activation="tanh"), initialization="uniform")
ann.add_layer(dense_layer(1, activation="sigmoid"), initialization="uniform")

ann.train(X_train,
          Y_train,
          lr=1e-5,
          epochs=1000000,
          loss="binary_crossentropy",
          plot_freq=1000)

pred = ann.predict(x_test)
ann.evaluate(pred, y_test)
