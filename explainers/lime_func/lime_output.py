import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def lime_func(model,train_data,feature_names,target_feature,instance_sequence,num_features=6):

    # transfer dataframe to np.array
    train_data = np.array(train_data)
    instance=train_data[instance_sequence]

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


    if plot:
        # import matplotlib.pyplot as plt
        # plt.barh(lime_df['feature'],lime_df['lime_val'])
        plt.show()
    else:
        pass


    return lime_df

def lime_explainations(model,train_data,features,
                       target,instance_sequence,
                       num_features=10,
                       plot=True,save=True,save_path='lime.jpg',**kwargs):
    '''

    :param model: sklearn model object
    :param train_data:dataframe, input feature dataset
    :param features:list, feature names
    :param target:string, target feature name
    :param instance_sequence:int, instance number
    :param num_features:int, number of features
    :param plot:bool, if plt.show()
    :param save:bool, if save the picture
    :param save_path:string, path of picture saved, default='lime.jpg'

    :return:dataframe, lime values
    '''
    # transfer dataframe to np.array
    train_data = np.array(train_data)
    instance = train_data[instance_sequence]

    # lime initialization
    predict_fn = lambda x: model.predict(x).astype(float)

    # Create the LIME Explainer

    explainer = lime.lime_tabular.LimeTabularExplainer(train_data, mode="regression", feature_names=features,
                                                       class_names=target, discretize_continuous=True)
    exp = explainer.explain_instance(instance, predict_fn, num_features=num_features)


    if plot:
        fig=exp.as_pyplot_figure()
        fig.show()
        plt.show()


    if save:
        fig.savefig(save_path)

    lime_list = exp.as_list()
    lime_df = pd.DataFrame(columns=["feature", "feature_upper_val", "feature_lower_val", "lime_val"])

    for index, lime_cont in enumerate(lime_list):

        feature_val = []
        for j in lime_cont[0].split():
            if _containenglish(j):
                feature = j
            if _is_number(j):
                feature_val.append(j)
        lime_df.loc[index] = (feature, max(feature_val), min(feature_val), lime_cont[1])

    return lime_df


