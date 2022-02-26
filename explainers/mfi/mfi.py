
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def _mse_feature_importance(model,data,target,preserve=True):

    standar=["<class 'sklearn.ensemble.forest.RandomForestRegressor'>",
             "<class 'sklearn.ensemble._forest.RandomForestRegressor'>"]
    str_type=str(type(model))
    # print(str_type)

    if str_type in standar:
        # check the model whether randomforest
        data=data.drop(target,axis=1)
        df=pd.DataFrame({"feature":data.columns[1:],
                     "MFI":model.feature_importances_[:-1]}) 
        if preserve:
            df.to_csv("mfi.csv")
        return df
    else:
        print("The model used is not tree-based, the MFI can not be estimated.")
        return None






def mse_feature_importance_plot(df,top=20,preserve=True,preserve_path='mfi.jpg',show=False):

    df_sort=df.sort_values(by="MFI",ascending=False)
    if len(df_sort)<top:
        df_top=df_sort
    else:
        df_top=df_sort[0:top]

    plt.figure()
    plt.barh(df_top["feature"],df_top["MFI"])
    plt.title('MSE-based feature importance')
    if preserve==True:
        plt.savefig(preserve_path)
    elif preserve==False:
        print('preserve==False, only for windows users')
        if show==False:
            pass
        else:
            plt.show()

def mse_feature_importance(model,data,target,plot=True,top=20,save=True,save_path='mfi.jpg'):
    '''

    :param model: sklearn model object, trained model
    :param data: pd.Dataframe, input data
    :param target:string, predicted target column name
    :param plot:bool, if plt.show()
    :param top:int, number of top feature at list
    :param save:bool, if save the picture
    :param save_path:string, path of picture saved
    :return:
    df: pd.Dataframe, MFI of features
    '''

    standar=["<class 'sklearn.ensemble.forest.RandomForestRegressor'>",
             "<class 'sklearn.ensemble._forest.RandomForestRegressor'>"]
    str_type=str(type(model))
    # print(str_type)

    if str_type in standar:
        # check the model whether randomforest
        data=data.drop(target,axis=1)
        df=pd.DataFrame({"feature":data.columns[1:],
                     "MFI":model.feature_importances_[:-1]})

        df_sort = df.sort_values(by="MFI", ascending=False)
        if len(df_sort) < top:
            df_top = df_sort
        else:
            df_top = df_sort[0:top]

        fig=plt.figure()
        plt.barh(df_top["feature"], df_top["MFI"])
        plt.title('MSE-based feature importance')
        if plot:
            plt.show()
        if save:
            fig.savefig(save_path)
        return df

    else:
        print("The model used is not tree-based, the MFI can not be estimated.")
        return None
