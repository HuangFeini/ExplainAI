import pandas as pd
from ..data_processing.data_processing_main import data_processing_main
import os
import sys

def input_dataset(flag=0):
    '''

    :param flag: index of which data set, see ExplainAI tutorials
    :return:read data input, pandas.Dataframe
    '''
    if flag==0:


        path=os.path.dirname(sys.modules['ExplainAI'].__file__)
        file_name=os.path.join(path,'flx_data/','dataset.csv')
        data = pd.read_csv(file_name)



    elif flag==1:

        path = os.path.dirname(sys.modules['ExplainAI'].__file__)
        file_name = os.path.join(path, 'flx_data/', 'dataset_process.csv')

        data = pd.read_csv(file_name)


    elif flag==2:
        # data_processing parameters are default.
        path = os.path.dirname(sys.modules['ExplainAI'].__file__)
        file_name = os.path.join(path, 'flx_data/', 'FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv')


        data = pd.read_csv(file_name,header=0)

        d = data_processing_main(data=data,
                                 target='SWC_F_MDS_1',
                                 time_add=1,
                                 lag_add=1,
                                 elim_SM_nan=1,
                                 drop_ir=1,
                                 drop_nan_feature=1,
                                 part=0.7,
                                 n_estimator=10,
                                 sbs=True,
                                 split='2')
        data, ss = d.total()
    return data


