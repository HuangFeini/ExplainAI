import pandas as pd
import matplotlib.pyplot as plt


#=========data processing

# from ExplainAI.data_processing.add_variables import time_add
# file='.\\flx_data\\FLX_CN-Ha2_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv'
# data0=pd.read_csv(file,header=0)
# new_data=time_add(data0)
#
#
# from ExplainAI.data_processing.add_variables import lag_add
# new_data=lag_add(new_data,sm_lag=7,p_lag=7)
#
#
# from ExplainAI.data_processing.data_cleaning import data_cleaning
# c1=data_cleaning(new_data)
# d1=c1.elim_SM_nan()
#
# c2=data_cleaning(d1)
# d2=c2.drop_ir()
#
# c3=data_cleaning(d2)
# d3=c3.drop_nan_feature()
#
#
#
# # from ExplainAI.data_processing.feature_selection import feature_selection
# # fs=feature_selection(data=new_data,target="SWC_F_MDS_1")
# # sbs_result=fs.sbs_rf(n_estimators=100)
# # print(sbs_result)
#
# from ExplainAI.data_processing.data_processing_main import data_processing_main
# d=data_processing_main(data=data0,
#                        target='SWC_F_MDS_1',
#                        time_add=True,
#                        lag_add=True,
#                        elim_SM_nan=True,
#                        drop_ir=True,
#                        drop_nan_feature=True,
#                        part=0.7,
#                        n_estimator=200,
#                        sbs=True,
#                        split=2)
# dd,ss=d.total()
# dd.to_csv("dd.csv")
# #dd is new_dataset after data processing
# ss.to_csv('ss.csv')
#======================================================================
# if run in Linux, use the following code
# plt.switch_backend('agg')
# file="/flx_data/dataset.csv" #your path
# d=pd.read_csv(file)
# print(d)

from ExplainAI.flx_data.input import input_dataset
d=input_dataset(flag=0)
# print(d)

from ExplainAI.data_processing.split_data import split_data
xtr,ytr,xte,yte=split_data(d,target="SWC").split_xy() #"SWC_F_MDS_1"
# print(xtr)
#
#
#
from ExplainAI.model.make_model import make_model
m,res,y_predict=make_model(modeltype='RandomForest',
                             x_train=xtr,
                             y_train=ytr,
                             x_test=xte,
                             y_test=yte)
print(res)


# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
# from sklearn import ensemble
# m=ensemble.RandomForestRegressor(n_estimators=500)
# m.fit(xtr,ytr)
# y_predict = m.predict(xte)
# res={"r2":r2_score(y_predict,yte),
#              "MSE":mean_squared_error(y_predict,yte),
#              "MAE":mean_absolute_error(y_predict,yte),
#              "RMSE":mean_squared_log_error(y_predict,yte)}
# print(res)







#------------1
# from ExplainAI.preview import info_plots
# import matplotlib.pyplot as plt
# # # show distribution with feature of interest ("TS")
# fig1, axes, summary_df = info_plots.actual_plot(model=m, X=xte, feature="TS", feature_name="TS")
# #
# fig2, axes, summary_df = info_plots.target_plot(df=d, target="SWC", feature="TS", feature_name="TS")
# # # show distribution under two features' interaction
# fig3, axes, summary_df = info_plots.actual_plot_interact(model=m, X=xte, features=["DOY", "TS"], feature_names=["DOY", "TS"])
# #
# fig4, axes, summary_df = info_plots.target_plot_interact(df=d, target="SWC", features=["DOY", "TS"], feature_names=["DOY", "TS"])
# plt.show()
# fig4.savefig('fig4.jpg')
# print(summary_df)

#
from ExplainAI.utils import get_x,get_features

x=get_x(d,target="SWC")
f=get_features(x)
#
# #------------pi ok
# from ExplainAI.explainers.pi.pi import permutation_importance
# p=permutation_importance(model=m, features=f, save=True,plot=True)
# print(p)

# #------------mfi ok
# from ExplainAI.explainers.mfi.mfi import mse_feature_importance
# mfi=mse_feature_importance(model=m, data=d, target="SWC",top=15,save=True,plot=True)
# print(mfi)


# # #------------ale ok
# from ExplainAI.explainers.ale.ale import accumulated_local_effect_1d,accumulated_local_effect_2d
# a1=accumulated_local_effect_1d(model=m, train_set=x, features='TS',plot=False,save=True,monte_carlo=False)
#
# a2=accumulated_local_effect_2d(m, train_set=x, features=['TS', 'DOY'], plot=False, bins=40,save=True)
# print(a2)

# # #------------ice ok
# from ExplainAI.explainers.ice.ice import individual_conditional_exception
# i=individual_conditional_exception(data=x, feature='TS', model=m,plot=True,save=True,save_path='ice.jpg')
# i.to_csv('ice.csv')


# #------------lime ok
# from ExplainAI.explainers.lime_func.lime_output import lime_explainations
# lime=lime_explainations(m, train_data=x, features=f, target="TS", instance_sequence=2,num_features=len(f),
#                         plot=True,save=True,save_path='lime.jpg')
# print(lime)



#------------pdp ok
# from ExplainAI.explainers.pdp.pdp import partial_dependence_plot_1d,partial_dependence_plot_2d
# pd1=partial_dependence_plot_1d(model=m,data=x,model_features=f,feature="TS",plot=True,save=True)
# print(pd1)
# #
# pd2=partial_dependence_plot_2d(model=m,data=x,model_features=f,features=["TS",'DOY'],plot=True,save=True)
# print(pd2)




#------------?
# from ExplainAI.explainers.shap_func.shap_func import shap_func
#
# ss = shap_func(m, x,f)
# ss.record_shap()
# ss.single_shap(nth=6)
# ss.feature_value_shap()
# # ss.time_shap()
# # plt.show()
# # #
# # #
# # ss.depend_shap(depend_feature='TS')
# #
# # ss.mean_shap()
# # ss.intera_shap()

#-------- ok
# from ExplainAI.explainers.shap_func.shap_func import shap_explainations_instance
# s=shap_explainations_instance(m,x,f,nth=4)
# print(s)

#-------- ok
# from ExplainAI.explainers.shap_func.shap_func import shap_explainations_mean
# sm=shap_explainations_mean(m,x,f)
# print(sm)

#-------- ok
# from ExplainAI.explainers.shap_func.shap_func import shap_explainations_time
# st=shap_explainations_time(m,x,f)

#-------- ok
# from ExplainAI.explainers.shap_func.shap_func import shap_explainations_dependence
# shap_explainations_dependence(m,x,f,dependence_feature='TS')


from ExplainAI.explainers.shap_func.shap_func import shap_explainations_feature_value
shap_explainations_feature_value(m,x,f)