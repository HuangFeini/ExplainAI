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
        file_name=os.path.join(path,'flx_data/','dataset.csv')
        data = pd.read_csv(file_name,header=0)



    return data


