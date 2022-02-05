from explainers.ice import ice
from model.randomforest_gv import randomforest
import matplotlib.pyplot as plt
if __name__ == '__main__':
    m,d,r2 = randomforest(re_feature_selection=False)
    d = d.drop("SWC", axis=1)
    ice_obj=ice.ice(data=d,column="TS",predict=m.predict)
    icep=ice.ice_plot(ice_obj,
                      column="TS",
                      plot_points=True,
                      color_by=None,
                      plot_pdp=True)
    plt.show()