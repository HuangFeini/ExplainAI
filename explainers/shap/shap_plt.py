import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def shap_time_xai(sv, features,plot=True, save=True,
                  describe=False, save_path='shap_time.jpg'):

    sv0 = sv[:, :-1]
    sv0 = pd.DataFrame(sv0, columns=features)

    df = sv0
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
        print(sv0.describe())

    return sv0


def shap_instance_xai(sv, features, nth, plot=True, save=True, save_path='shap_instance.jpg'):
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

    sv0 = sv[:, :-1]
    s = pd.DataFrame(sv0, columns=features)

    left = 0
    right = 0
    fig, ax = plt.subplots(figsize=(10, 2))
    for index in range(len(features)):

        if s.iloc[nth, index] < 0:
            plt.barh(0, s.iloc[nth, index], 0.1, label=features[index], left=left)
            left = -abs(s.iloc[nth, index]) + left

        if s.iloc[index, nth] > 0:
            plt.barh(0, s.iloc[nth, index], 0.1, label=features[index], left=right)
            right = abs(s.iloc[nth, index]) + right

    title = 'Shapley values of individual instance     ' + 'nth=' + str(nth)
    plt.title(title)
    plt.legend(ncol=len(features))

    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)

    if plot:
        plt.show()
    if save:
        fig.savefig(save_path)

    return s.iloc[nth, :]


def shap_mean_xai(sv, features, plot=True, save=True, describe=False, save_path='shap_mean.jpg'):
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

    sv0 = sv[:, :-1]
    s = pd.DataFrame(sv0, columns=features)


    ds = pd.DataFrame({'features': features, 'Shapley_mean': s.mean()})

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

def shap_dependence_xai(sv,m,x, features, dependence_feature,plot=True, save=True, describe=False,save_path='shap_dependence.jpg'):
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


    sv0 = sv[:, :-1]
    s = pd.DataFrame(sv0, columns=features)


    sf=s[dependence_feature]
    xf=x[dependence_feature]
    yp=m.predict(x)

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


def shap_feature_xai(sv,x,features, plot=True, save=True, describe=False,save_path='shap_feav.jpg'):
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


    sv0 = sv[:, :-1]
    s = pd.DataFrame(sv0, columns=features)


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


def _color_class_0(df, num=4):
    if num == 2:
        c = []
        for i in df:
            if i > df.mean():
                c.append('r')
            else:
                c.append('g')
        return c
    if num == 4:
        c = []
        div = (df.max() - df.min()) / num
        for i in df:
            if (df.min() + div) > i > df.min():
                c.append('green')
            elif (df.min() + div * 2) > i > (df.min() + div):
                c.append('blue')
            elif (df.min() + div * 3) > i > (df.min() + div * 2):
                c.append('yellow')
            elif (df.min() + div * 4) > i > (df.min() + div * 3):
                c.append('red')
        return c

def _color_class(df, num=2):
    df0=np.array(df)
    if num == 2:
        c = []
        for i in df0:
            if i > df0.mean():
                c.append('r')
            else:
                c.append('g')
        return c
    if num == 4:
        c = []
        div = (df0.max() - df0.min()) / num
        for i in df:
            if (df0.min() + div) > i > df0.min():
                c.append('green')
            elif (df0.min() + div * 2) > i > (df0.min() + div):
                c.append('blue')
            elif (df0.min() + div * 3) > i > (df0.min() + div * 2):
                c.append('yellow')
            elif (df0.min() + div * 4) > i > (df0.min() + div * 3):
                c.append('red')

    return c

def _index_df(index_df, x, feature):
    index_list = index_df.to_list()
    # print(type(index_df))
    # print(index_df)
    xf = x[feature]
    feature_values = []
    for index in index_list:
        feature_value = xf.iloc[index]
        feature_values.append(feature_value)

    return feature_values


def shap_feature_value_d_xai(sv, x,  features, subplot=True, color_num=2,plot=True, save=True, describe=False,save_path='shap_feav.jpg'):
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

    sv0 = sv[:, :-1]
    s = pd.DataFrame(sv0, columns=features)

    features.append('legend')


    if subplot:
        fig=plt.figure(figsize=(10,5))
        i=1
        for feature in features:
            ax = fig.add_subplot(1, len(features), i)
            if i<len(features):
                sf = pd.DataFrame({'feature': [feature] * len(s[feature]), 'VALUE': s[feature]})
                shap_p = list(sf['VALUE'])
                shap_p.sort()
                shap_sort_index = sf.VALUE.argsort()
                kernel = st.gaussian_kde(shap_p)
                K = kernel(shap_p)

                feature_value=_index_df(shap_sort_index,x,feature)
                c = _color_class(feature_value, num=color_num)

                ax.hlines(y=shap_p, xmin=0, xmax=list(K), colors=c)
                ax.set_title(feature)
                i = i + 1
            elif i==(len(features)):
                if color_num==2:
                    ax.plot([0,0], [0, 0.5], linewidth=10,color='green')
                    ax.plot([0,0], [0.5, 1], linewidth=10, color='red')
                    ax.text(0, -0.05, 'low')
                    ax.text(0, 1.05,'high')
                    ax.axis('off')
                    ax.set_xticks([])
                    ax.set_yticks([])
                elif color_num==4:
                    ax.plot([0,0], [0, 0.25], linewidth=10,color='green')
                    ax.plot([0,0], [0.25, 0.5], linewidth=10, color='blue')
                    ax.plot([0, 0], [0.5, 0.75], linewidth=10, color='yellow')
                    ax.plot([0, 0], [0.75, 1], linewidth=10, color='red')
                    ax.text(0.05, -0.05, 'low')
                    ax.text(0.05, 0.2, 'mid-low')
                    ax.text(0.05, 0.45, 'mid')
                    ax.text(0.05, 0.7, 'mid-high')
                    ax.text(0.05, 0.95,'high')
                    ax.axis('off')
                    ax.set_xticks([])
                    ax.set_yticks([])
        fig.subplots_adjust(hspace=-5)
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 5))
        i = 1
        for feature in features:
            ax = fig.add_subplot(1, len(features), i)
            if i < len(features):
                sf = pd.DataFrame({'feature': [feature] * len(s[feature]), 'VALUE': s[feature]})
                shap_p = list(sf['VALUE'])
                shap_p.sort()
                shap_sort_index = sf.VALUE.argsort()
                kernel = st.gaussian_kde(shap_p)
                K = kernel(shap_p)

                feature_value = _index_df(shap_sort_index, x, feature)
                c = _color_class(feature_value, num=color_num)
                ax.hlines(y=shap_p, xmin=0, xmax=list(K), colors=c)
                ax.set_title(feature)

                ax.set_ylim(-15,15)
                # ax.set_xlim(0, 30)
                ax.set_xticks([])


                if i==1:
                    ax.spines['right'].set_color('white')
                elif i==len(features):
                    ax.spines['left'].set_color('white')
                    ax.set_yticks([])
                else:
                    ax.spines['left'].set_color('white')
                    ax.spines['right'].set_color('white')
                    ax.set_yticks([])


                i = i + 1
            elif i == (len(features)):
                if color_num == 2:
                    ax.plot([0, 0], [0, 0.5], linewidth=10, color='green')
                    ax.plot([0, 0], [0.5, 1], linewidth=10, color='red')
                    ax.text(0, -0.05, 'low')
                    ax.text(0, 1.05, 'high')
                    ax.axis('off')
                    ax.set_xticks([])
                    ax.set_yticks([])
                elif color_num == 4:
                    ax.plot([0, 0], [0, 0.25], linewidth=10, color='green')
                    ax.plot([0, 0], [0.25, 0.5], linewidth=10, color='blue')
                    ax.plot([0, 0], [0.5, 0.75], linewidth=10, color='yellow')
                    ax.plot([0, 0], [0.75, 1], linewidth=10, color='red')
                    ax.text(0.05, -0.05, 'low')
                    ax.text(0.05, 0.2, 'mid-low')
                    ax.text(0.05, 0.45, 'mid')
                    ax.text(0.05, 0.7, 'mid-high')
                    ax.text(0.05, 0.95, 'high')
                    ax.axis('off')
                    ax.set_xticks([])
                    ax.set_yticks([])
        plt.subplots_adjust(hspace=-10)
        plt.show()

        if plot:
            plt.show()
        if save:
            fig.savefig(save_path)

        sxk=pd.DataFrame({'shap_sort_index':shap_sort_index,'K':K,'feature_value':feature_value,'shapley':shap_p})
        sxk=sxk.sort_values(by="shap_sort_index", ascending=True)


        if describe:
            sxk.describe()

        return sxk

