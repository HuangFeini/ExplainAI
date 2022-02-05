from feature_selection.feature_selection import SplitData
import pandas as pd
from sklearn import tree, svm
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn import neighbors,ensemble
from feature_selection.feature_selection import obtain_select_data
from utils import _get_site_folder


def select_best_model_regression(f,i,preserve=True,re_feature_selection=False):
    '''

    :param data:
    :return:

    '''
    if re_feature_selection:
        data = obtain_select_data(f, i)

    else:
        data_file = f +"\\"+"new_desc_sele_data.csv"
        site_path=pd.read_csv(data_file)["SITE_PATH"][i]
        data_path=site_path+"\\data_confirm.csv"
        data=pd.read_csv(data_path)


    # ss = preprocessing.StandardScaler()
    train, test = SplitData(data).Split()

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



    dt,dt_r2=func(tree.DecisionTreeRegressor(), x_train, y_train, x_test, y_test)
    lr,lr_r2=func(linear_model.LinearRegression(), x_train, y_train, x_test, y_test)
    kn,kn_r2=func(neighbors.KNeighborsRegressor(), x_train, y_train, x_test, y_test)
    rf,rf_r2=func(ensemble.RandomForestRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
    abr,abr_r2=func(ensemble.AdaBoostRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
    gbr,gbr_r2=func(ensemble.GradientBoostingRegressor(n_estimators=100), x_train, y_train, x_test, y_test)
    bg,bg_r2=func(ensemble.BaggingRegressor(), x_train, y_train, x_test, y_test)
    bys,bys_r2=func(linear_model.BayesianRidge(), x_train, y_train, x_test, y_test)
    sv,sv_r2=func(svm.SVR(), x_train, y_train, x_test, y_test)

    abb_list=['DecisionTree','Linear','KNeighbors','RandomForest','AdaBoost','GradientBoosting','Bagging','BayesianRidge','SVR']
    r2_list=[dt_r2,lr_r2,kn_r2,rf_r2,abr_r2,gbr_r2,bg_r2,bys_r2,sv_r2]
    model_list=[dt,lr,kn,rf,abr,gbr,bg,bys,sv]

    model_r2=pd.DataFrame({"model_name":abb_list,"r2":r2_list,"model_obj":model_list})

    model_best_index=model_r2[model_r2["r2"]==model_r2["r2"].max()].index.tolist()[0]
    model_best=model_list[model_best_index]


    if preserve:
        site_path=_get_site_folder(f,i)
        model_r2.to_csv(site_path+"\\model_r2.csv")


    return model_best, data




