import pandas as pd
import numpy as np


class UnlimitedDataWorks:

    def __init__(self):
        print("Starting Reality Marble...")
        self.count = 0

    def train_test_split(self, df, normalize=False):
        data = pd.DataFrame([])
        if normalize:
            for col in df.columns:
                mx = df[col].max()
                mn = df[col].min()
                df[col] = (df[col] - mn) / (mx - mn)

        # generate a 70-30 split on the data:
        X = df[["lat", "lon"]][:304113]
        Y = df["alt"][:304113]
        x = df[["lat", "lon"]][304113:]
        y = df["alt"][304113:]
        return (np.array(X), np.array(Y), np.array(x), np.array(y))
