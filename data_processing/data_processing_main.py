import pandas as pd
from ..data_processing.add_variables import time_add,lag_add
from ..data_processing.split_data import split_data
from ..data_processing.data_cleaning import data_cleaning
from ..data_processing.feature_selection import feature_selection

class data_processing_main():
    def __init__(self, data,time_add,lag_add,elim_SM_nan, drop_ir, drop_nan_feature,part,n_estimator,sbs):
        self.data=data
        self.time_add=time_add
        self.lag_add=lag_add
        self.elim_SM_nan=elim_SM_nan
        self.drop_ir=drop_ir
        self.drop_nan_feature=drop_nan_feature
        self.part=part
        self.n_estimator=n_estimator
        self.sbs=sbs





    def total(self):


        if self.time_add:
            self.data = time_add(self.data)

        if self.lag_add:
            self.data=lag_add(self.data)

        if self.elim_SM_nan:
            fa1=data_cleaning(self.data)
            self.data=fa1.elim_SM_nan()
        if self.drop_ir:
            fa2 = data_cleaning(self.data)
            self.data=fa2.drop_ir()
        if self.drop_nan_feature:
            fa3=data_cleaning(self.data)
            self.data=fa3.drop_nan_feature()
        if self.sbs:
            sd=split_data(self.data,part=self.part)
            train,test=sd.split()
            fb=feature_selection(train)

            feature_sequence=fb.sbs_rf(n_estimators=self.n_estimator)

            return self.data,feature_sequence
        else:
            return self.data




# if __name__ == '__main__':
#
#     file='E:\\xai\\flx_data\\FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'
#     data=pd.read_csv(file,header=0)
#
#     d=data_processing_main(data=data,
#                           time_add=1,
#                            lag_add=1,
#                           elim_SM_nan=1,
#                           drop_ir=1,
#                           drop_nan_feature=1,
#                           part=0.7,
#                           n_estimator=10,
#                           sbs=True)
#     dd,ss=d.total()
#     dd.to_csv("dd.csv")
#     ss.to_csv('ss.csv')