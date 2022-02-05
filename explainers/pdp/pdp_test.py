import matplotlib.pyplot as plt
from model.randomforest_gv import randomforest
from explainers.pdp import pdp


if __name__ == '__main__':

    #1.modeling
    m,d,r2 = randomforest(re_feature_selection=False)
    da = d.drop("SWC", axis=1)

    #2.one-dimentional PDP object
    pdp1=pdp.pdp_isolate(model=m,
                         dataset=da,
                         model_features=da.columns,
                         feature="TS")
    #2.1.PDP plot
    fig3, axes =pdp.pdp_plot(pdp1,"TS")
    plt.show()
    #2.2.obtain PDP result as dataframe
    print(pdp1.count_data)

    #3.two-dimentional PDP object
    pdp2=pdp.pdp_interact(model=m,
                          dataset=da,
                          model_features=da.columns,
                          features=["TS","DOY"])
    # 3.1.PDP plot
    fig4, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp2,
        feature_names=["TS","DOY"],
        plot_pdp='contour')
    #3.2.obtain PDP result as dataframe
    print(pdp2.pdp)
    plt.show()
