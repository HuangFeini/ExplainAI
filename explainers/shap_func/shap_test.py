from model.randomforest_gv import randomforest
from explainers.shap_func.shap_func import shap_func
import matplotlib.pyplot as plt
if __name__ == '__main__':


    m,d,r2 = randomforest(re_feature_selection=False)

    x = d.drop("SWC", axis=1)
    y=d["SWC"]

    ss=shap_func(m,x)

    #
    # ss.single_shap(nth=6)
    #
    # ss.feature_value_shap()
    #
    ss.time_shap()
    # plt.show()
    #
    #
    # ss.depend_shap(depend_feature='TS')
    #
    # ss.mean_shap()
    # ss.intera_shap()