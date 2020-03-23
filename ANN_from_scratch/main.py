from Model import nn_sequential_model
from Layers import dense_layer
from DataPreprocessor import UnlimitedDataWorks
import pandas as pd


columns = ["junk", "lat", "lon", "alt"]
raw_df = pd.read_csv("data.txt",
                     sep=',',
                     header=None,
                     names=columns).drop("junk", 1).sample(frac=1)

pre_processor = UnlimitedDataWorks()
X_train, Y_train, x_test, y_test = pre_processor.train_test_split(raw_df,
                                                                  normalize=True)

ann = nn_sequential_model()
ann.add_layer(dense_layer(2, activation="linear"))
# ann.add_layer(dense_layer(5, activation="sigmoid"))
# ann.add_layer(dense_layer(4, activation="sigmoid"))
ann.add_layer(dense_layer(1, activation="linear"))
ann.train(X_train, Y_train, lr=5e-6, epochs=1001, plot_freq=10)
pred = ann.predict(x_test)
ann.evaluate(pred, y_test)
