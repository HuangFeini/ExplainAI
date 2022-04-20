import pandas as pd
import os
import sys

def input_dataset(flag=0):
    '''

    :param flag: index of which data set, see ExplainAI tutorials
    :return:read data input, pandas.Dataframe
    '''
    if flag==0:


        path=os.path.dirname(sys.modules['ExplainAI'].__file__)
        file_name=os.path.join(path,'flx_data/','dataset0.csv')
        data = pd.read_csv(file_name)



    elif flag==1:

        path = os.path.dirname(sys.modules['ExplainAI'].__file__)
        file_name = os.path.join(path, 'flx_data/', 'dataset1.csv')

        data = pd.read_csv(file_name)



    return data


