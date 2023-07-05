
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

def permutation_importance_xai(m,f,x,y, plot=True,save=True,save_path='pi.jpg', seed=None, n_repeats=5):#x是训练集的,y是对应的
    result={}
    from sklearn.inspection import permutation_importance
    i = permutation_importance(m, x, y, random_state=seed, n_repeats=n_repeats)
    result['feature'] = f
    result['pi'] = i.importances_mean
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by='pi', ascending=False)
    print(result_df)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.barh(result_df['feature'][::-1],result_df['pi'][::-1])
    plt.title('Permutation Feature Importance')


    if plot:
        plt.show()

    if save:
        fig.savefig(save_path)


    return result_df




