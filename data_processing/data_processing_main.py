
from ..data_processing.add_variables import time_add,lag_add
from ..data_processing.split_data import split_data
from ..data_processing.data_cleaning import data_cleaning
from ..data_processing.feature_selection import feature_selection

class data_processing_main():
    '''
    the integrated procedures for data processing.
    `:parameter`

    `data:` pd.Dataframe,input data

    `target:` string, predicted target column name

    `time_add:` bool, whether time_add

    `lag_add:`bool, whether lag_add

    `elim_SM_nan:` bool, whether elim_SM_nan

    `drop_ir:` eliminate data of irrelevant records in FLUXNET,like percentiles, quality index, RANDUNC, se, sd...
    `drop_nan_feature:`Eliminate the features with too many(30%) Nan.
    `part:`part of split_data
    `n_estimator:`sequential backward selection using random forest, n_estimator of random forest
    `sbs:`whether use sbs

    `split:` int 2 or int 3, if 2, splite data to training set and testing set; if 3, training set, validating set and testing set
    '''
    def __init__(self, data,target,time_add,lag_add,elim_SM_nan, drop_ir, drop_nan_feature,part,n_estimator,sbs,split):
        self.data=data
        self.time_add=time_add
        self.lag_add=lag_add
        self.elim_SM_nan=elim_SM_nan
        self.drop_ir=drop_ir
        self.drop_nan_feature=drop_nan_feature
        self.part=part
        self.n_estimator=n_estimator
        self.sbs=sbs
        self.target=target
        self.split=split


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
            sd=split_data(self.data,target=self.target,part=self.part)
            if self.split == str(2) or int(2):
                train,test=sd.split()
            elif self.split == str(3) or int(3):
                train, valid, test = sd.split3()
            fb=feature_selection(train,target=self.target)

            feature_sequence=fb.sbs_rf(n_estimators=self.n_estimator)

            return self.data,feature_sequence
        else:
            return self.data



