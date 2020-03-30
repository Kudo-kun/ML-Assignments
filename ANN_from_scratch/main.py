import pandas as pd
import numpy as np
from Layers import dense_layer
from Model import nn_sequential_model
from DataPreprocessor import UnlimitedDataWorks


columns = ["F2", "F3", "F4", "T"]
raw_df = pd.read_csv("data1.txt",
                     sep=',',
                     header=None,
                     names=columns).drop("F2", 1).sample(frac=1)

pre_processor = UnlimitedDataWorks()
X_train, Y_train, x_test, y_test = pre_processor.train_test_split(raw_df,
																  xfeatures=2,
																  normalize=True)

ann = nn_sequential_model()
ann.add_layer(dense_layer(2, activation="linear"))
ann.add_layer(dense_layer(5, activation="tanh"))
ann.add_layer(dense_layer(4, activation="tanh"))
ann.add_layer(dense_layer(1, activation="linear"))

ann.train(X_train,
          Y_train,
          lr=5e-5,
          epochs=1000000,
          loss="MSE",
          plot_freq=500)

pred = ann.predict(x_test)
ann.evaluate(pred, y_test)
