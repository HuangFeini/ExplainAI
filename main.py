
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #-----------------测试用的数据和模型----------------------start-----------
    import pandas as pd
    import numpy as np
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    data=pd.read_csv('dataset.csv',header=0)
    X=data.iloc[:,1:data.shape[1]]
    Y=data.iloc[:,0]
    f=X.columns

    X_train = data.iloc[0:round(data.shape[0]*0.7),1:data.shape[1]]
    X_test = data.iloc[round(data.shape[0]*0.7)+1:data.shape[0],1:data.shape[1]]
    Y_train = data.iloc[0:round(data.shape[0]*0.7),0]
    Y_test = data.iloc[round(data.shape[0]*0.7)+1:data.shape[0],0]


    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    pre_test = rf.predict(X_test)

    RMSE=mean_squared_error(pre_test,Y_test)
    R2=r2_score(pre_test,Y_test)

    print(R2)

    # plt.scatter(Y_test,pre_test)
    # plt.show()


    # -----------------测试用的数据和模型----------------------over-----------

    import shap

    explainer = shap.TreeExplainer(rf)
    shap_value = explainer.shap_values(X_test)
    print(shap_value)
    shap.summary_plot(shap_value, X_test)














    #----------------------ALE 测试  bug  0927--------------start----------
    #1.一维ALE图，可以画图，保存数据有点问题!!!!!
    # from ale import ale
    # ale.ale_plot(model=rf, train_set=X, features="TS")
    #2.二维ALE图，可以画图，保存数据有点问题!!!!! 画图结果和R语言的不一样！！！！！
    # from ale import ale
    # ale.ale_plot(model=rf, train_set=X_train, features=["TS","SW"],bins=2)
    #----------------------ALE 测试  bug  0927---------------over----------






    #----------------------ice 测试  成功  0927--------------start----------
    # from ice import ice
    # ice_obj=ice.ice(data=X_test, column="TS", predict=rf.predict)
    # icep=ice.ice_plot(ice_obj,plot_points=True,color_by=None)
    # plt.show()

    #----------------------ice 测试  成功  0927-------------- over----------





    #---------------------pdp 测试 成功 0927-----------------start----------

    # from pdp import pdp, info_plots
    #1.提供预测值的分布
    # fig1, axes, summary_df= info_plots.actual_plot(
    #     model=rf, X=X, feature="TA",feature_name="TA")
    #2.提供实际值的分布
    # fig2, axes, summary_df= info_plots.target_plot(
    #     df=data, target="SWC", feature="TA",feature_name="TA")
    #3.一维pdp图
    # pdp1=pdp.pdp_isolate(model=rf, dataset=X_train, model_features=f, feature="TS")
    # fig3, axes =pdp.pdp_plot(pdp1,"TS")
    #4.保存一维PDP数据
    # print(pdp_data.count_data) # pdp数据，dataFrame
    #5.二维PDP图
    # pdp2=pdp.pdp_interact(model=rf, dataset=X, model_features=f, features=["TS","SW"])
    # fig4, axes = pdp.pdp_interact_plot(
    #     pdp_interact_out=pdp2, feature_names=["TS","SW"], plot_pdp='contour')
    #6.保存二维PDP数据
    # print(pdp2.pdp) # pdp数据，dataFrame

    # plt.show()

    # ---------------------pdp 测试 成功 0927-----------------over----------












