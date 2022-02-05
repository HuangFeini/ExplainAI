from model.randomforest_gv import randomforest_gv
import pandas as pd
from utils import _get_version_res_folder
import matplotlib.pyplot as plt


def mse_feature_importance(model,data,preserve=True):

    standar=["<class 'sklearn.ensemble.forest.RandomForestRegressor'>",
             "<class 'sklearn.ensemble._forest.RandomForestRegressor'>"]
    str_type=str(type(model))
    print(str_type)

    if str_type in standar:
        # check the model whether randomforest
        data=data.drop("SWC",axis=1)
        df=pd.DataFrame({"feature":data.columns[1:],
                     "MFI":model.feature_importances_[:-1]}) 
        if preserve:
            df.to_csv("mfi.csv")

        return df




def mse_feature_importance_plot(df):

    df1=df.sort_values(by="MFI",ascending=True)
    plt.figure()
    plt.barh(df1["feature"],df1["MFI"])
    plt.show()


