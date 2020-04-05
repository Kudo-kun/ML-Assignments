import numpy as np
import pandas as pd
import DataPreprocessor as dp
from Model import LogisticRegression


columns = ["F1", "F2", "F3", "F4", "T"]
raw_df = pd.read_csv("data.txt", header=None, names=columns, sep=",")
raw_df = raw_df.sample(n=len(raw_df), random_state=11)
X_train, Y_train, xval, yval, x_test, y_test = dp.train_test_split(raw_df,
                                                       			   normalize=False,
                                                       			   standardize=True)

model = LogisticRegression(nfeatures=4)
model.compile(epochs=5000, 
			  learning_rate=1e-3, 
			  penalty="L2",
			  metrics="fscore")

model.fit(X_train,
		  Y_train,
		  xval=xval,
		  yval=yval,
		  plot_freq=None)

pred = model.predict(x_test, model.weights)
model.evaluate(pred, y_test, verbose=True)