import pandas as pd
import numpy as np


def train_test_split(df, split_ratio, normalize=False, standardize=False):
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

	# generate a split on the data:
	sz = int(len(df) * split_ratio)
	xfeatures=len(df.columns) - 1
	xattr = df.columns[:xfeatures]
	yattr = df.columns[xfeatures:]
	X = df[xattr][:sz]
	Y = df[yattr][:sz]
	x = df[xattr][sz:]
	y = df[yattr][sz:]
	return (np.array(X), np.array(Y), np.array(x), np.array(y))
