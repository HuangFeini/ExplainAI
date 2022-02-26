


class split_data():
    '''
    Split dataset to training set and testing set.
    According to the time-sequence (using data of time-ahead to predict feature data).
    ------
    :parameter
    data[dataframe]: A data set
    part[float]: division proportion for two default as 0.7
    part3[list]: A list division proportion for three, default as [0.7,0.2,0.1]

    :returns
    dataset of train, test
    '''
    def __init__(self,data,target,part=0.7,part3=[0.7,0.2,0.1]):
        self.data=data
        self.part=part
        self.part3=part3
        self.target=target

    def split(self):
        length=self.data.shape[0]
        split=round(length*self.part)
        train=self.data[0:split]
        test=self.data[split:length]

        return train,test


    def split3(self):
        length=self.data.shape[0]
        split1=round(length*self.part3[0])
        train=self.data[0:split1]

        split2=round(length*self.part3[1]+split1)
        valid=self.data[split1:split2]

        test=self.data[split2:length]


        return train, valid, test

    def split_xy(self):
        length=self.data.shape[0]
        split=round(length*self.part)
        train=self.data[0:split]
        test=self.data[split:length]
        y_train=train[self.target]
        x_train=train.drop(self.target,axis=1)
        y_test=test[self.target]
        x_test=test.drop(self.target,axis=1)
        return x_train,y_train, x_test,y_test