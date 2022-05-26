
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def permutation_importance(estimator,feature_names=None,plot=True,save=True,save_path='pi.jpg'):
    """
    Return an explanation of a tree-based ensemble estimator.
    """
    coef = estimator.feature_importances_
    trees = np.array(estimator.estimators_).ravel()
    coef_std = np.std([tree.feature_importances_ for tree in trees], axis=0)

    pi=pd.DataFrame({'features':feature_names,'weights':coef,'std':coef_std})

    fig=plt.figure()
    plt.barh(pi['features'], pi['weights'])
    plt.title("Permutation importance")

    if plot:
        plt.show()

    if save:
        fig.savefig(save_path)

    return pi

def permutation_importance_xai(m,f,x,y,rmse,plot=True,save=True,save_path='pi.jpg'):#x是训练集的,y是对应的
    result=[]
    for feauture in f:
        x_scramble=x.copy()
        x_scramble[feauture]=x[feauture].sample(frac=1).values
        from sklearn.metrics import mean_squared_error
        y_scramble=m.predict(x_scramble)
        rmse_scramble=mean_squared_error(y_scramble, y)
        result.append({'feature':feauture,'pi':abs(rmse-rmse_scramble)})
    # result_df=pd.DataFrame(result).sort_values(by='pi',ascending=True)
    result_df = pd.DataFrame(result)
    print(result_df)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.barh(result_df['feature'],result_df['pi'])
    plt.title('Permutation Feature Importance')


    if plot:
        plt.show()

    if save:
        fig.savefig(save_path)


    return result_df




