import pandas as pd

import time
start = time.time()


d=pd.read_csv(r'D://codes//ExplainAI//ExplainAI//flx_data//dataset0.csv',header=0)

from ExplainAI.utils import get_x,get_features
x=get_x(d,target="SWC")
f=get_features(x)
y_ob=d['SWC']
y=y_ob

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y_ob,test_size=0.33)

from ExplainAI.model.make_model import make_model
m,res,y_predict=make_model(modeltype='RandomForest',
                             x_train=xtr,
                             y_train=ytr,
                             x_test=xte,
                             y_test=yte)
print(res)


from ExplainAI.preview import info_plots
import matplotlib.pyplot as plt
# # show distribution with feature of interest ("TS")
fig1, axes, summary_df = info_plots.actual_plot(model=m, X=xte, feature="TS", feature_name="TS")
#
fig2, axes, summary_df = info_plots.target_plot(df=d, target="SWC", feature="TS", feature_name="TS")
# # show distribution under two features' interaction
fig3, axes, summary_df = info_plots.actual_plot_interact(model=m, X=xte, features=["DOY", "TS"], feature_names=["DOY", "TS"])
#
fig4, axes, summary_df = info_plots.target_plot_interact(df=d, target="SWC", features=["DOY", "TS"], feature_names=["DOY", "TS"])
plt.show()
fig4.savefig('fig4.jpg')
print(summary_df)

# pi
from ExplainAI.explainers.pi.pi import permutation_importance_xai
rmse=res['RMSE']
permutation_importance_xai(m,f,x,y,rmse)

#lime
from ExplainAI.explainers.lime.lime_xai import lime_xai
lime_res=lime_xai(m=m,x=x,y_ob=y_ob,instance=5,n=10000,num_bins=25)
print(lime_res)

# #mfi
from ExplainAI.explainers.mfi.mfi import mse_feature_importance
mfi=mse_feature_importance(model=m, data=d, target="SWC",top=15,save=True,plot=True)
print(mfi)
#
#ale
from ExplainAI.explainers.ale.ale import accumulated_local_effect_1d,accumulated_local_effect_2d
a1=accumulated_local_effect_1d(model=m, train_set=x, features='TS',plot=False,save=True,monte_carlo=False)
a2=accumulated_local_effect_2d(m, train_set=x, features=['TS', 'DOY'], plot=False, bins=40,save=True)
print(a2)

# #ice
from ExplainAI.explainers.ice.ice import individual_conditional_exception
i=individual_conditional_exception(data=x, feature='TS', model=m,plot=True,save=True,save_path='ice.jpg')
#
# #pdp
from ExplainAI.explainers.pdp.pdp import partial_dependence_plot_1d,partial_dependence_plot_2d
pd1=partial_dependence_plot_1d(model=m,data=x,model_features=f,feature="TS",plot=True,save=True)
print(pd1)
pd2=partial_dependence_plot_2d(model=m,data=x,model_features=f,features=["TS",'DOY'],plot=True,save=True)
print(pd2)

#shapley

from ExplainAI.explainers.shap.shap_xai import TreeExplainer
ex = TreeExplainer(m)

sv = ex.shap_values(x)
from ExplainAI.explainers.shap.shap_plt import shap_time_xai
shap_time_xai(sv,f,plot=True, save=True,
                            describe=True,save_path='shap_time.jpg')
from ExplainAI.explainers.shap.shap_plt import shap_instance_xai
shap_instance_xai(sv,f,5,plot=True,save=True,save_path='shap_instance.jpg')

from ExplainAI.explainers.shap.shap_plt import shap_mean_xai
shap_mean_xai(sv, f, plot=True, save=True, describe=False, save_path='shap_mean.jpg')


from ExplainAI.explainers.shap.shap_plt import shap_dependence_xai
shap_dependence_xai(sv,m,x, f, dependence_feature='TS',plot=True, save=True, describe=False)

from ExplainAI.explainers.shap.shap_plt import shap_feature_xai
shap_feature_xai(sv,x=x,features=f, plot=True, save=True, describe=False,save_path='shap_feav.jpg')

#没有降雨没值
from ExplainAI.explainers.shap.shap_plt import shap_feature_value_d_xai
shap_feature_value_d_xai(sv, x=x,features=f, subplot=True, color_num=2,plot=True, save=True)

print('time:',time.time() - start)














