
class data_cleaning():
    '''
    Data washing.
    :parameter
    data: pd.Dataframe,input data
    :returns
    Newdata: pd.Dataframe, output data
    '''
    def __init__(self,data):
        self.data=data

    def elim_SM_nan(self):
        '''
        Eliminate the observation without SM values.
        '''
        length = self.data.shape[0]
        SM = self.data["SWC_F_MDS_1"].values.tolist()
        nanNum = SM.count(-9999)
        nanLimit = 0.3 * length
        if nanNum > nanLimit:
            print("NaN in SM exceed the limit of 30%, please choose another site.")
            return None
        else:
            NewData=self.data[-self.data["SWC_F_MDS_1"].isin([-9999])]
            return NewData

    def drop_ir(self):
        '''
        Eliminate irrelevant records in FLUXNET,
        like percentiles, quality index, RANDUNC, se, sd...
        '''
        EZList=['JOINTUNC','QC','SE','SD', 'RANDUNC','_05','_16','_25','_75','_50','_95','_84']
        FeaList = self.data.columns.tolist()
        DropFeaList=[]
        for i in FeaList:
            for j in EZList:
                if j in i:
                    DropFeaList.append(i)

        NewData=self.data.drop(DropFeaList,axis=1)

        return NewData

    def drop_nan_feature(self):
        '''
        Eliminate the features with too many(30%) Nan.
        '''
        length = self.data.shape[0]
        nanLimit = 0.3 * length
        DropNanList=[]

        for f in self.data.columns.tolist():
            value=self.data[f].values.tolist()
            nanNum = value.count(-9999)
            if nanNum > nanLimit:
                DropNanList.append(f)

        NewData = self.data.drop(DropNanList, axis=1)

        return NewData