import pandas as pd
import numpy as np
from Layers import dense_layer
from Model import nn_sequential_model
from DataPreprocessor import UnlimitedDataWorks


columns = ["F1", "F2", "F3", "F4", "T"]
raw_df = pd.read_csv("data2.txt",
                     sep=',',
                     header=None,
                     names=columns).sample(frac=1)

pre_processor = UnlimitedDataWorks()
X_train, Y_train, x_test, y_test = pre_processor.train_test_split(raw_df,
                                                                  xfeatures=4,
                                                                  normalize=False)

ann = nn_sequential_model()
ann.add_layer(dense_layer(4, activation="linear"))
ann.add_layer(dense_layer(5, activation="tanh"))
ann.add_layer(dense_layer(4, activation="sigmoid"))
ann.add_layer(dense_layer(1, activation="sigmoid"))

ann.train(X_train,
          Y_train,
          lr=5e-5,
          epochs=1000000,
          loss="binary_crossentropy",
          plot_freq=1000)

pred = ann.predict(x_test)
ann.evaluate(pred, y_test)
