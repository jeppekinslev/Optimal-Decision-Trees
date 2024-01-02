# author: Jeppe

import pandas as pd
from ucimlrepo import fetch_ucirepo


def loadData(dataname):
    """
    load dataset
    """
    df = fetch_ucirepo(name=dataname)
    x = df.data.features
    y = df.data.targets
    # drop NA values
    x = x.dropna(axis=0)
    y = y.iloc[x.index]
    #facotrize x and y
    x = x.apply(lambda a: pd.factorize(a)[0])
    y = y.apply(lambda a: pd.factorize(a)[0])
    # Min-Max Normalize features to interval [0,1] if max =! 0 for each feature
    for i in range(x.shape[1]):
        if x.iloc[:,i].max() != 0:
            x.iloc[:,i] = (x.iloc[:,i] - x.iloc[:,i].min()) / (x.iloc[:,i].max() - x.iloc[:,i].min())
    # Convert to numpy array
    x = x.to_numpy()
    y = y.values.flatten()
    return x, y