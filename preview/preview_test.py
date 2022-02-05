import matplotlib.pyplot as plt
from model.randomforest_gv import randomforest

from preview import info_plots



if __name__ == '__main__':
    # f = "D:\\SM data\\FLUXNET"
    # m,d = randomforest_gv(f,0,version="v2",only_rf=True,preserve=False)
    # print(d.shape)
    # da = d.drop("SWC_F_MDS_1", axis=1)

    m,d,r2 = randomforest(re_feature_selection=False)
    # print(d.shape)
    da = d.drop("SWC", axis=1)



    # 1.提供预测值的分布
    fig1, axes, summary_df = info_plots.actual_plot(
        model=m, X=da, feature="TS", feature_name="TS")

    # 2.提供实际值的分布
    fig2, axes, summary_df= info_plots.target_plot(
        df=d, target="SWC", feature="TS",feature_name="TS")

    fig3, axes, summary_df = info_plots.actual_plot_interact(
        model=m,X=da, features=["DOY","TS"], feature_names=["DOY","TS"])

    fig4, axes, summary_df = info_plots.target_plot_interact(
        df=d, target="SWC", features=["DOY","TS"], feature_names=["DOY","TS"])

    plt.show()