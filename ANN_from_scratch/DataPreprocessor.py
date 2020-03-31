import pandas as pd
import numpy as np


def train_test_split(df, xfeatures, split_ratio, normalize=False):
    data = pd.DataFrame([])
    if normalize:
        for col in df.columns:
            mx = df[col].max()
            mn = df[col].min()
            df[col] = (df[col] - mn) / (mx - mn)

    # generate a split on the data:
    sz = int(len(df) * split_ratio)
    xattr = df.columns[:xfeatures]
    yattr = df.columns[xfeatures:]
    X = df[xattr][:sz]
    Y = df[yattr][:sz]
    x = df[xattr][sz:]
    y = df[yattr][sz:]
    return (np.array(X), np.array(Y), np.array(x), np.array(y))
