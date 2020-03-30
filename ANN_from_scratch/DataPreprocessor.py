import pandas as pd
import numpy as np


class UnlimitedDataWorks:

    def __init__(self):
        print("Starting Reality Marble...")

    def train_test_split(self, df, xfeatures, normalize=False):
        data = pd.DataFrame([])
        if normalize:
            for col in df.columns:
                mx = df[col].max()
                mn = df[col].min()
                df[col] = (df[col] - mn) / (mx - mn)

        # generate a 70-30 split on the data:
        sz = int(len(df) * 0.7)
        xattr = df.columns[:xfeatures]
        yattr = df.columns[xfeatures:]
        X = df[xattr][:sz]
        Y = df[yattr][:sz]
        x = df[xattr][sz:]
        y = df[yattr][sz:]
        return (np.array(X), np.array(Y), np.array(x), np.array(y))
