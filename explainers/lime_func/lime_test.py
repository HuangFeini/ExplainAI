import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from feature_selection.feature_selection import split_data
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from explainers.lime_func.lime_output import _obtain_lime,lime_output



if __name__ == '__main__':
    f = "D:\\SM data\\FLUXNET"
    data_file = f + "\\" + "new_desc_sele_data_origin.csv"
    site_path = pd.read_csv(data_file)["SITE_PATH"][0]
    site_name =pd.read_csv(data_file)["SITE_ID"][0]
    print(0, site_path ,site_name)

    # read input data
    data_path = site_path + "\\data_confirm.csv"
    data = pd.read_csv(data_path, index_col=0)
    cols =data.columns
    print(cols)


    # Handling missing values
    imputer_mean= SimpleImputer(missing_values=-9999, strategy='mean')
    for col in cols:
        data[col] = imputer_mean.fit_transform(data[[col]])

    # Divide into test and train:
    train, test = split_data(data).split()
    x_train = np.array(train.drop("SWC_F_MDS_1", axis=1))
    y_train = np.array(train["SWC_F_MDS_1"])
    x_test = np.array(test.drop("SWC_F_MDS_1", axis=1))
    y_test = np.array(test["SWC_F_MDS_1"])

    x = data.drop("SWC_F_MDS_1", axis=1)
    y = data["SWC_F_MDS_1"]




    # random forest modelling
    model =RandomForestRegressor(n_estimators=100)
    model.fit(x_train ,y_train)
    y_predict =model.predict(x_test)
    r2 =r2_score(y_test ,y_predict)


    feature_names =cols.drop("SWC_F_MDS_1")


    exp =_obtain_lime(model ,train_data=x_train ,feature_names=feature_names ,target_feature="P_F" ,instance=x_test[0])
    ins ,out =lime_output(exp ,plot=False)

    out.to_csv("out.csv")


