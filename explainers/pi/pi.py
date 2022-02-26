import eli5
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def pi_trans(model,feature_names,preserve):

    # # Permutation Importance
    perm = eli5.sklearn.explain_rf_feature_importance(estimator=model,feature_names=feature_names)

    pfi_str=str(perm.feature_importances)[32:-16]
    pfi_list=pfi_str.split('),')

    feature_list=[]
    weight_list=[]
    std_list=[]

    for i in pfi_list:
        feature_list.append(str(i[i.index("feature=")+len("feature=")+1:i.index("weight=")-3]))
        weight_list.append(float(i[i.index("weight=")+len("weight="):i.index("std=")-2]))
        std_list.append(float(i[i.index("std=")+len("std="):i.index("value=")-2]))

    pfi_pd=pd.DataFrame({"feature":feature_list,"weight":weight_list,"std":std_list})


    if preserve:
        pfi_pd.to_csv("pfi.csv")

    return pfi_pd


def pi_plot(pi_pd,preserve=True,preserve_path='pi.jpg',show=False):
    plt.figure()
    plt.barh(pi_pd['feature'],pi_pd['weight'])
    plt.title("Permutation importance")
    if preserve==True:
        plt.savefig(preserve_path)
    elif preserve==False:
        print('preserve==False, only for windows users')
        if show==False:
            pass
        else:
            plt.show()


def permutation_importance(model,features,plot=True,save=True,save_path='pi.jpg'):
    '''
    # # Permutation Importance
    :param model: sklearn model object, trained model
    :param features: list or turple, fearture names storaged in a list or a turple
    :param plot: bool, if plt.show()
    :param save: bool, if save the picture
    :param save_path: string, path of picture saved
    :return:
    pfi_pd: pd.Dataframe, PI of features
    '''

    perm = eli5.sklearn.explain_rf_feature_importance(estimator=model,feature_names=features)

    pfi_str=str(perm.feature_importances)[32:-16]
    pfi_list=pfi_str.split('),')

    feature_list=[]
    weight_list=[]
    std_list=[]

    for i in pfi_list:
        feature_list.append(str(i[i.index("feature=")+len("feature=")+1:i.index("weight=")-3]))
        weight_list.append(float(i[i.index("weight=")+len("weight="):i.index("std=")-2]))
        std_list.append(float(i[i.index("std=")+len("std="):i.index("value=")-2]))

    pfi_pd=pd.DataFrame({"feature":feature_list,"weight":weight_list,"std":std_list})

    fig=plt.figure()
    plt.barh(pfi_pd['feature'], pfi_pd['weight'])
    plt.title("Permutation importance")

    if plot:
        plt.show()

    if save:
        fig.savefig(save_path)

    return pfi_pd



