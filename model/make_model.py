from sklearn import tree, svm
# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error #,mean_squared_log_error
from sklearn import linear_model
from sklearn import neighbors,ensemble
import numpy as np
def make_model(modeltype, x_train, y_train, x_test, y_test):
    def func(clf,x_train,y_train,x_test,y_test):
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)
        res={"r2":r2_score(y_predict,y_test),
             "MSE":mean_squared_error(y_predict,y_test),
             "MAE":mean_absolute_error(y_predict,y_test),
             "RMSE":np.sqrt(mean_squared_error(y_predict,y_test))}

        return clf,res,y_predict

    model_list = ['DecisionTree', 'Linear', 'KNeighbors',
                  'RandomForest', 'AdaBoost',
                  'GradientBoosting', 'Bagging',
                  'BayesianRidge', 'SVR']

    if modeltype=='DecisionTree':
        dt,dt_res,y_predict=func(tree.DecisionTreeRegressor(), x_train, y_train, x_test, y_test)
        return dt,dt_res,y_predict

    elif modeltype=='Linear':
        lr,lr_res,y_predict=func(linear_model.LinearRegression(), x_train, y_train, x_test, y_test)
        return lr, lr_res, y_predict

    elif modeltype=='KNeighbors':
        kn,kn_res,y_predict=func(neighbors.KNeighborsRegressor(), x_train, y_train, x_test, y_test)
        return kn,kn_res,y_predict
    elif modeltype=='RandomForest':
        rf,rf_res,y_predict=func(ensemble.RandomForestRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
        return rf,rf_res,y_predict

    elif modeltype == 'AdaBoost':
        abr,abr_res,y_predict=func(ensemble.AdaBoostRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
        return abr,abr_res,y_predict

    elif modeltype == 'GradientBoosting':
        gbr,gbr_res,y_predict=func(ensemble.GradientBoostingRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
        return gbr,gbr_res,y_predict

    elif modeltype == 'Bagging':
        bg,bg_res,y_predict=func(ensemble.BaggingRegressor(), x_train, y_train, x_test, y_test)
        return bg,bg_res,y_predict

    elif modeltype == 'BayesianRidge':
        bys,bys_res,y_predict=func(linear_model.BayesianRidge(), x_train, y_train, x_test, y_test)
        return bys,bys_res,y_predict

    elif modeltype == 'SVR':
        sv,sv_res,y_predict=func(svm.SVR(), x_train, y_train, x_test, y_test)
        return sv,sv_res,y_predict
    else:
        print("please check your model type again")





