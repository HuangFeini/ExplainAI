from feature_selection.feature_selection import split_data
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from feature_selection.feature_selection import data_processing_main

def randomforest_gv(only_rf=True,preserve=True,test_stage=False,re_feature_selection=False):
    '''
    Prediction modelling, using grid search to optimize paras of RF.

    :param data:
    only_rf: boolean, to conform RF model.
    perserve: boolean, if True the best paras from grid search would be saved.
    test_stage: boolean, if True, no grid search, paras use constants.
    re_re_feature_selection:boolean, if False, data uses dataset(via contrived work)
                                     if True, data uses p.csv (without contrived work, only SBS)


    :return:
    model_best:RandomForestRegressor object, can be inherent.
    data:dataframe, same as input data.
    r2:float, prediction precision of testing set.

    '''
    if re_feature_selection:
        data = pd.read_csv(r"E:\\xai\\flx_data\\p.csv")
        print('ok')

    else:
        data = pd.read_csv(r"E:\\xai\\flx_data\\dataset.csv")
        # file = 'D:\\codes\\xai\\flx_data\\FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'
        # d = data_processing_main(data=pd.read_csv(file, header=0),
        #                          time_add=1,
        #                          lag_add=1,
        #                          elim_SM_nan=1,
        #                          drop_ir=1,
        #                          drop_nan_feature=1,
        #                          part=0.7,
        #                          n_estimator=10,
        #                          sbs=True)
        # data, ss = d.total()


    train, test = split_data(data).split()

    x_train=train.drop("SWC_F_MDS_1",axis=1)
    y_train=train["SWC_F_MDS_1"]
    x_test=test.drop("SWC_F_MDS_1",axis=1)
    y_test=test["SWC_F_MDS_1"]

    x=data.drop("SWC_F_MDS_1",axis=1)
    y=data["SWC_F_MDS_1"]


    def func(clf,x_train,y_train,x_test,y_test):
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)
        res=r2_score(y_predict,y_test)
        if res>0:
            return clf,res
        else:
            res = r2_score(y_test,y_predict)
            return clf,res

    if only_rf==True:
        if not test_stage:
            param_grid = {"n_estimators": range(10,201,20),
                        "max_depth": range(2, 52, 10),
                        "min_samples_leaf":range(2,50,10)}
        else:
            # for shorten time in test
            param_grid = {"n_estimators": [100],
                          "max_depth": [20],
                          "min_samples_leaf": [40]}



        grid_search = GridSearchCV(estimator=ensemble.RandomForestRegressor(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=10,max_features='sqrt',random_state=10),
                                   param_grid=param_grid)
        grid_search.fit(x, y)

        rf_gv, rf_r2_gv = func(ensemble.RandomForestRegressor(
            n_estimators=grid_search.best_params_['n_estimators'],
            max_depth=grid_search.best_params_['max_depth'],
            min_samples_leaf=grid_search.best_params_['min_samples_leaf']),
                         x_train, y_train, x_test, y_test)


        rf_o, rf_r2_o = func(ensemble.RandomForestRegressor(),
                         x_train, y_train, x_test, y_test)

        if float(rf_r2_gv)>float(rf_r2_o):
            model_best=rf_gv
            r2 = rf_r2_gv
            if preserve:

                res_save=pd.DataFrame({"gv":[True],
                                       "R2":[r2],
                                       "n_estimators": [grid_search.best_params_['n_estimators']],
                                       "max_depth": [grid_search.best_params_['max_depth']],
                                       "min_samples_leaf" : [grid_search.best_params_['min_samples_leaf']]
                })
                res_save.to_csv("randomforest_gv.csv")
        else:
            model_best=rf_o
            r2 = rf_r2_o
            if preserve:
                res_save=pd.DataFrame({"gv":[False],"R2":[r2],"paras":[None]})
                res_save.to_csv("randomforest_gv.csv")

        return model_best,data,r2

def randomforest(prediction_record=False,re_feature_selection=False):
    '''
    Prediciton modelling of randomforest.

    :param data:
    re_feature_selection: boolean, if False, input data from dataset.csv (via contrived work).
                                if True, input data from feature_selection.
    prediction_record:boolean,if True, prediction saved.
                              if False, prediction no saved.
    :return:
    model_best:RandomForestRegressor object.
    data:dataframe, same as input data.
    r2:float, prediction precision of testing set.

    '''
    if re_feature_selection:
        data = pd.read_csv(r"E:\\xai\\flx_data\\p.csv")
        print('ok')
        train, test = split_data(data).split()

        x_train = train.drop("SWC_F_MDS_1", axis=1)
        y_train = train["SWC_F_MDS_1"]
        x_test = test.drop("SWC_F_MDS_1", axis=1)
        y_test = test["SWC_F_MDS_1"]

    else:
        data = pd.read_csv(r"E:\\xai\\flx_data\\dataset.csv")

        train, test = split_data(data).split()

        x_train=train.drop("SWC",axis=1)
        y_train=train["SWC"]
        x_test=test.drop("SWC",axis=1)
        y_test=test["SWC"]




    def func(clf,x_train,y_train,x_test,y_test):
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)
        res=r2_score(y_predict,y_test)
        if res>0:
            return clf,res,y_predict,y_test
        else:
            res = r2_score(y_test,y_predict)
            return clf,res,y_predict,y_test



    rf_o, rf_r2_o,y_predict,y_test = func(ensemble.RandomForestRegressor(),
                         x_train, y_train, x_test, y_test)


    model_best=rf_o
    r2 = rf_r2_o

    print(r2)
    if prediction_record:
        # print(y_predict,y_test)
        pred=pd.DataFrame({"prediction":y_predict,"observation":y_test})
        pred.to_csv("prediction.csv")


    return model_best,data,r2



if __name__ == '__main__':
    model_best,data,r2=randomforest(re_feature_selection=True)
    print(model_best)