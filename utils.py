import pandas as pd


def get_x(data,target):
    x=data.drop(target,axis=1)
    return x

def get_features(x):
    features=list(x.columns)
    return features





