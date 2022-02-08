import pandas as pd
# ----155555
# file="./flx_data/dataset.csv"
# # d=pd.read_csv(file)
# # print(d)

from flx_data.input import input_dataset
d=input_dataset(flag=0)
# print(d)

from data_processing.split_data import split_data
xtr,ytr,xte,yte=split_data(d,target="SWC").split()
# print(xtr)



from model.make_model import make_model
m,res,y_predict=make_model(modeltype='GradientBoosting',
                             x_train=xtr,
                             y_train=ytr,
                             x_test=xte,
                             y_test=yte)
print(res)

#------------
from preview import info_plots
import matplotlib.pyplot as plt
# show distribution with feature of interest ("TS")
fig1, axes, summary_df = info_plots.actual_plot(model=m, X=xte, feature="TS", feature_name="TS")

fig2, axes, summary_df = info_plots.target_plot(df=d, target="SWC", feature="TS", feature_name="TS")
# show distribution under two features' interaction
fig3, axes, summary_df = info_plots.actual_plot_interact(model=m, X=xte, features=["DOY", "TS"], feature_names=["DOY", "TS"])

fig4, axes, summary_df = info_plots.target_plot_interact(df=d, target="SWC", features=["DOY", "TS"], feature_names=["DOY", "TS"])
# plt.show()


from utils import get_x,get_features

x=get_x(d,target="SWC")
f=get_features(x)

#------------
# from explainers.pi.pi import pi_trans,pi_plot
# p = pi_trans(model=m, feature_names=list(da.columns), preserve=False)
# pi_plot(p)

#------------
from explainers.ale.ale import ale_plot
import matplotlib.pyplot as plt
from explainers.ale.ale_output import ale_output,ale_plot_total
ale_plot(model=m, train_set=x, features='TS',plot=True)


ale_plot(m, train_set=x, features=tuple(['TS', 'DOY']), plot=False, bins=40, monte_carlo=False)
plt.show()
ale_plot_total(model=m, data=x)
ale_output(model=m, data=x,preserve=False)

#------------
from explainers.ice import ice
ice_obj=ice.ice(data=x,column="TS",predict=m.predict)
icep=ice.ice_plot(ice_obj,
                      column="TS",
                      plot_points=True,
                      color_by=None,
                      plot_pdp=True)
plt.show()
#------------
from explainers.lime_func.lime_output import lime_func,lime_output
import numpy as np
x = np.array(x)
exp = lime_func(m, train_data=x, feature_names=f, target_feature="TS", instance=x[0])
ins, out = lime_output(exp, plot=True)

#------------
# from explainers.pdp import pdp
# pdp1 = pdp.pdp_isolate(model=m,
#                        dataset=x,
#                        model_features=f,
#                        feature="TS")
# # 2.1.PDP plot
# fig3, axes = pdp.pdp_plot(pdp1, "TS")
# plt.show()
# # 2.2.obtain PDP result as dataframe
# print(pdp1.count_data)
#
# # 3.two-dimentional PDP object
# pdp2 = pdp.pdp_interact(model=m,
#                         dataset=x,
#                         model_features=f,
#                         features=["TS", "DOY"])
# # 3.1.PDP plot
# fig4, axes = pdp.pdp_interact_plot(
#     pdp_interact_out=pdp2,
#     feature_names=["TS", "DOY"],
#     plot_pdp='contour')
# # 3.2.obtain PDP result as dataframe
# print(pdp2.pdp)
# plt.show()

#------------
# from explainers.shap_func.shap_func import shap_func
#
# ss = shap_func(m, x)
#
# #
# # ss.single_shap(nth=6)
# #
# ss.feature_value_shap()
# #
# # ss.time_shap()
# plt.show()
# #
# #
# ss.depend_shap(depend_feature='TS')
#
# ss.mean_shap()
# ss.intera_shap()