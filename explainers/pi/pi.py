
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

def permutation_importance_xai(m,f,x,y,rmse):#x是训练集的,y是对应的
    result=[]
    for feauture in f:
        x_scramble=x.copy()
        x_scramble[feauture]=x[feauture].sample(frac=1).values
        from sklearn.metrics import mean_squared_error
        y_scramble=m.predict(x_scramble)
        rmse_scramble=mean_squared_error(y_scramble, y)
        result.append({'feature':feauture,'pi':abs(rmse-rmse_scramble)})
    result_df=pd.DataFrame(result).sort_values(by='pi',ascending=True)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_ylabel('Increase in RMSE')
    ax.set_xlabel('Predictor')
    ax.set_title('Permutation Feature Importance')

    predictors = result_df.feature
    y_pos = range(len(predictors))
    scores = result_df.pi
    ax.bar(predictors, scores)
    plt.xticks(y_pos, predictors, rotation=45)
    plt.show()


    return result_df




