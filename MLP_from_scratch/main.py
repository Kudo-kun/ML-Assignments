import pandas as pd
import numpy as np
from Layers import dense_layer
from Model import nn_sequential_model
import DataPreprocessor as dp


columns = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "T"]
raw_df = pd.read_csv("data.txt", header=None, names=columns, sep=",")
raw_df = raw_df.sample(n=len(raw_df), random_state=11)
X_train, Y_train, x_test, y_test = dp.train_test_split(raw_df,
                                                       split_ratio=0.8,
                                                       normalize=True)

ann = nn_sequential_model()
ann.add_layer(dense_layer(10, activation="relu"))
ann.add_layer(dense_layer(5, activation="relu"))
ann.add_layer(dense_layer(5	, activation="relu"))
ann.add_layer(dense_layer(1, activation="sigmoid"))

ann.compile(learning_rate=0.01,
            epochs=40000,
            loss="binary_crossentropy")

ann.fit(X_train, Y_train, plot_freq=None)

print("training metrics:")
ann.evaluate(ann.predict(X_train), Y_train)
print("\ntesting metrics:")
ann.evaluate(ann.predict(x_test), y_test)
