import pandas as pd
from sklearn import ensemble

class feature_selection():
    '''
    Using sequential backward selection (SBS) to select optimal features.
    Based on random forest (RF).
    :parameter
    data: dataframe
    target: string, name of target
    :returns
    NFeaFrame：dataframe, including features and their importance calculated by MSE.

    '''
    def __init__(self,data,target):
        self.data=data
        self.target=target

    def sbs_rf(self,n_estimators):


        FeaList=self.data.columns.tolist()
        if self.target in FeaList:
            FeaList.remove(self.target)
            x = self.data.drop(self.target, axis=1)
        else:
            x = self.data

        NFeaFrame=pd.DataFrame(columns=['ElimFeature','score'])

        for i in range(1,self.data.shape[1]):
            RF=ensemble.RandomForestRegressor(n_estimators=n_estimators)
            y=self.data[self.target]
            RF.fit(x,y)
            score=RF.score(x,y)

            fi=RF.feature_importances_
            fii={'Feature':x.columns.tolist(), 'Importance':fi}
            fiFrame=pd.DataFrame(fii)


            minFea=fiFrame['Feature'][fiFrame['Importance']==fiFrame['Importance'].min()]

            xm=minFea.tolist()[0]
            NFeaFrame.loc[i]=[xm,score]
            x=x.drop(minFea, axis=1)

        return NFeaFrame






















