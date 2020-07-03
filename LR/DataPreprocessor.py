import pandas as pd
import numpy as np


def train_test_split(df, normalize=False, standardize=False):
    for col in df.columns:
        if col != "T":
            if normalize:
                mx = df[col].max()
                mn = df[col].min()
                df[col] = (df[col] - mn) / (mx - mn)
            elif standardize:
                u = df[col].mean()
                sig = df[col].std()
                df[col] = (df[col] - u) / sig

    # generate a 70:10:20 split on the data:
    df.insert(0, "F0", np.ones(len(df)), True)
    xfeatures = len(df.columns) - 1
    sz1 = int(len(df) * 0.7)
    sz2 = int(len(df) * 0.1) + sz1
    xattr = df.columns[:xfeatures]
    yattr = df.columns[xfeatures:]
    
    X = df[xattr][:sz1]
    Y = df[yattr][:sz1]
    xval = df[xattr][sz1:sz2]
    yval = df[yattr][sz1:sz2]
    x = df[xattr][sz2:]
    y = df[yattr][sz2:]
    
    return (np.array(X), np.array(Y), np.array(xval), np.array(yval), np.array(x), np.array(y))
