import eli5
from model.randomforest_gv import randomforest
from explainers.pi.pi import pi_trans,pi_plot
if __name__ == '__main__':

    m,d,r2 = randomforest(re_feature_selection=False)
    da = d.drop("SWC", axis=1)
    p=pi_trans(model=m,feature_names=list(da.columns),preserve=False)
    pi_plot(p)
