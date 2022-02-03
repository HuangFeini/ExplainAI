import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from feature_selection.feature_selection import split_data
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

def _obtain_lime(model,train_data,feature_names,target_feature,instance,num_features=6):
    # lime initialization
    predict_fn = lambda x: model.predict(x).astype(float)

    # Create the LIME Explainer

    explainer = lime.lime_tabular.LimeTabularExplainer(train_data, mode="regression", feature_names=feature_names,
                                                       class_names=target_feature, discretize_continuous=True)
    exp = explainer.explain_instance(instance, predict_fn, num_features=num_features)

    return exp

def _containenglish(str0):
    import re
    return bool(re.search('[a-zA-Z]', str0))


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def lime_output(exp,plot=False):
    '''
    lime output for an instance, ragardless of which is the instance
    :param exp: a lime object from _obtain_lime
    :param plot: plot a bar picture for this instance
    :return:
    "feature", "feature_upper_val", "feature_lower_val", "lime_val"
    '''

    if plot:
        exp.as_pyplot_figure().show()

    lime_list=exp.as_list()
    lime_df = pd.DataFrame(columns=["feature", "feature_upper_val", "feature_lower_val", "lime_val"])


    for index,lime_cont in enumerate(lime_list):

        feature_val=[]
        for j in lime_cont[0].split():
            if _containenglish(j):
                feature = j
            if _is_number(j):
                feature_val.append(j)
        lime_df.loc[index] = (feature, max(feature_val), min(feature_val), lime_cont[1])

    return lime_df

