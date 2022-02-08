import eli5
import matplotlib.pyplot as plt
import pandas as pd



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


def pi_plot(pi_pd):
    plt.figure()
    plt.barh(pi_pd['feature'],pi_pd['weight'])
    plt.title("Permutation importance")
    plt.show()

