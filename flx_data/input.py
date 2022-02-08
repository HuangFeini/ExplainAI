import pandas as pd
from ..data_processing.data_processing_main import data_processing_main
import os
def input_dataset(flag=0):
    if flag==0:

        path=os.path.dirname(os.__file__)+"\site-packages\ExplainAI"
        file_name=path+r"\flx_data\dataset.csv"
        # print(file_name)
        data = pd.read_csv(file_name)



    elif flag==1:

        path=os.path.dirname(os.__file__)+"\site-packages\ExplainAI"
        file_name=path+r"./flx_data/dataset_process.csv"
        # print(file_name)
        data = pd.read_csv(file_name)


    elif flag==2:
        path=os.path.dirname(os.__file__)+"\site-packages\ExplainAI"
        file_name=path+r'./flx_data/FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'

        data = pd.read_csv(file_name,header=0)

        d = data_processing_main(data=data,
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


