import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
class shap_func():
    '''

    :param model: shap object
    :param x:input feature dataset
    :param features:list, feature names
    :param save_path:
    '''
    def __init__(self,model,x,features,save_path='shap.csv'):

        self.model=model
        self.explainer=shap.TreeExplainer(self.model)
        self.x=x
        self.shap_values=self.explainer.shap_values(self.x)
        self.save=True
        self.save_path=save_path
        self.features=features

    def record_shap(self):
        s = pd.DataFrame(self.shap_values,columns=self.features)
        if self.save:
            s.to_csv(self.save_path)
        return s

    def single_shap(self,nth):
        shap.force_plot(self.explainer.expected_value,
                        self.shap_values[nth, :],
                        self.x.iloc[nth, :],
                        matplotlib=False)

    def feature_value_shap(self):
        shap.summary_plot(self.shap_values, self.x)

    def time_shap(self):
        shap.force_plot(base_value=self.explainer.expected_value,
                        shap_values=self.shap_values,
                        features=self.x)

    def depend_shap(self,depend_feature):
        depend_feature=str(depend_feature)
        shap.dependence_plot(depend_feature, self.shap_values, self.x)

    def mean_shap(self):
        shap.summary_plot(self.shap_values, self.x, plot_type="bar")

    def intera_shap(self):
        shap_interaction_values = self.explainer.shap_interaction_values(self.x)
        shap.summary_plot(shap_interaction_values, self.x)


def shap_explainations_instance(model,x,features,nth,plot=True,save=True,save_path='shap_instance.jpg'):
    '''

    :param model: sklearn model object
    :param x:dataframe, input feature dataset
    :param features:list, feature names
    :param nth:int, sequence of instance of interest
    :param plot:bool, if plt.show()
    :param save:bool, if save the picture
    :param save_path:string, path of picture saved, default='pdp.jpg'

    :return: dataframe, shap valure of the instance
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    s = pd.DataFrame(shap_values, columns=features)


    left=0
    right=0
    fig,ax=plt.subplots(figsize=(10, 2))
    for index in range(len(features)):

        if s.iloc[nth,index]<0:
            plt.barh(0,s.iloc[nth,index],0.1,label=features[index],left=left)
            left=-abs(s.iloc[nth,index])+left

        if s.iloc[index,nth]>0:
            plt.barh(0,s.iloc[nth,index],0.1,label=features[index],left=right)
            right=abs(s.iloc[nth,index])+right

    title='Shapley values of individual instance     ' + 'nth='+ str(nth)
    plt.title(title)
    plt.legend(ncol=len(features))

    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)

    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)

    return s.iloc[nth,:]


def shap_explainations_mean(model, x, features, plot=True, save=True, describe=False,save_path='shap_mean.jpg'):
    '''
    :param model: sklearn model object
    :param x:dataframe, input feature dataset
    :param features:list, feature names
    :param plot:bool, if plt.show()
    :param save:bool, if save the picture
    :param save_path:string, path of picture saved, default='shap_mean.jpg'
    :param describe:bool, if True, the shapley values will be described statistically witl mean, std, min, max and so on.

    :return:
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    s = pd.DataFrame(shap_values, columns=features)


    ds=pd.DataFrame({'features':features,'Shapley_mean':s.mean()})


    fig = plt.figure()
    plt.barh(ds["features"], ds["Shapley_mean"])
    plt.title('Mean Shapley values')
    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)
    if describe:
        print(s.describe())

    return ds

def shap_explainations_time(model, x, features, plot=True, save=True, describe=False,save_path='shap_time.jpg'):
    '''

    :param model:
    :param x:
    :param features:
    :param plot:
    :param save:
    :param describe:
    :param save_path:
    :return:
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    s = pd.DataFrame(shap_values, columns=features)

    df=s
    fig, ax = plt.subplots()
    # split dataframe df into negative only and positive only values
    df_neg, df_pos = df.clip(upper=0), df.clip(lower=0)
    # stacked area plot of positive values
    df_pos.plot.area(ax=ax, stacked=True, linewidth=0.)
    # reset the color cycle
    ax.set_prop_cycle(None)
    # stacked area plot of negative values, prepend column names with '_' such that they don't appear in the legend
    df_neg.rename(columns=lambda x: '_' + x).plot.area(ax=ax, stacked=True, linewidth=0.)
    # rescale the y axis
    ax.set_ylim([df_neg.sum(axis=1).min(), df_pos.sum(axis=1).max()])
    plt.title('Shapley values with time series')


    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)
    if describe:
        print(s.describe())

    return s


def shap_explainations_dependence(model, x, features, dependence_feature,plot=True, save=True, describe=False,save_path='shap_dependence.jpg'):
    '''

    :param model:
    :param x:
    :param features:
    :param dependence_feature:
    :param plot:
    :param save:
    :param describe:
    :param save_path:
    :return:
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    s = pd.DataFrame(shap_values, columns=features)

    sf=s[dependence_feature]
    xf=x[dependence_feature]
    yp=model.predict(x)

    fig,ax= plt.subplots()
    plt.scatter(x=xf,y=sf,c=yp)

    plt.xlabel(str(dependence_feature))
    plt.ylabel('Shapley values for {f}'.format(f=str(dependence_feature)))
    cb=plt.colorbar()
    cb.set_label('Predicted target')
    plt.title('Dependent Shapley values')


    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)
    if describe:
        print(s.describe())

    return s


def shap_explainations_feature_value(model, x, features, plot=True, save=True, describe=False,save_path='shap_feav.jpg'):
    '''

    :param model:
    :param x:
    :param features:
    :param plot:
    :param save:
    :param describe:
    :param save_path:
    :return:
    '''

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    s = pd.DataFrame(shap_values, columns=features)




    def _minmax_norm(df):
        return (df - df.min()) / (df.max() - df.min())
    x = _minmax_norm(x)


    sx=pd.DataFrame()
    for feature in features:
        sf = s[feature]
        xf=x[feature]
        sf_sns=pd.DataFrame({'feature':[feature]*len(sf),'value':sf,'color':xf})
        sx=sx.append(sf_sns)

    fig=plt.figure(figsize=(15,5))
    plt.scatter(sx['feature'],sx['value'],c=sx['color'],marker='$▔▔▔▔▔▔$',s=1000,alpha=0.7)#——————
    plt.ylabel('Shapley values')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=0)
    cb.set_label('Feature values')
    plt.text(17,15,'High')
    plt.text(17,-15, 'Low')


    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)
    if describe:
        print(sx.describe())

    return sx