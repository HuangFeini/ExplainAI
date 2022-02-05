import pandas as pd
from data_processing.data_processing_main import data_processing_main

def input_dataset(flag=0):
    if flag==0:
        data = pd.read_csv(r"./flx_data/dataset.csv")
    elif flag==1:
        data = pd.read_csv(r"./flx_data/dataset_process.csv")
    elif flag==2:
        file = './flx_data/FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'
        d = data_processing_main(data=pd.read_csv(file, header=0),
                                 time_add=1,
                                 lag_add=1,
                                 elim_SM_nan=1,
                                 drop_ir=1,
                                 drop_nan_feature=1,
                                 part=0.7,
                                 n_estimator=10,
                                 sbs=True)
        data, ss = d.total()
    return data


