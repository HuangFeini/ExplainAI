
from explainers.lime_func.lime_output import _obtain_lime,lime_output
from model.randomforest_gv import randomforest


if __name__ == '__main__':


    m, d, r2 = randomforest(re_feature_selection=False)
    d = d.drop("SWC", axis=1)
    dfn=d.columns
    print(dfn)



    feature_names =dfn


    exp =_obtain_lime(m ,train_data=d ,feature_names=feature_names ,target_feature="TS" ,instance=d[0])
    ins ,out =lime_output(exp ,plot=False, instance=d[0])
    print(out)

    # out.to_csv("out.csv")


