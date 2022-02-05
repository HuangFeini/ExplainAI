from explainers.lime_func.lime_output import _obtain_lime,lime_output
import pandas as pd

def lime_multiply(data,model,train_data,feature_names,target_feature,num_features):
    i = 0
    for index in data.index:

        exp = _obtain_lime(model, train_data=train_data, feature_names=feature_names, target_feature=target_feature,
                           instance=data.loc[index], num_features=num_features)
        out = lime_output(exp, plot=False)
        out = out.drop(labels=["feature_upper_val", "feature_lower_val"], axis=1)
        out0 = out
        if i == 1:
            res = pd.merge(left=out0, right=out, how="outer", on="feature")
        if i > 1:
            res = pd.merge(left=res, right=out, how="outer", on="feature")
            # print(res)

            if i == len(data.index)-1:
                return res
        i = i + 1


