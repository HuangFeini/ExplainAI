
from model.randomforest_gv import randomforest
from explainers.mfi.mfi import mse_feature_importance,mse_feature_importance_plot



if __name__ == '__main__':
    m, d, r2 = randomforest(re_feature_selection=False)
    mfii=mse_feature_importance(model=m, data=d, preserve=False)
    print(mfii)
    mse_feature_importance_plot(mfii)
    print(r2)