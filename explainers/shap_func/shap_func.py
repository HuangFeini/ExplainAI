import shap
import pandas as pd
import numpy as np

class shap_func():
    def __init__(self,model,x):
        self.model=model
        self.explainer=shap.TreeExplainer(self.model)
        self.x=x
        self.shap_values=self.explainer.shap_values(self.x)


    def record_shap(self):
        s = pd.DataFrame(self.shap_values)
        s.to_csv('shap.csv')
        return s

    def single_shap(self,nth):
        shap.force_plot(self.explainer.expected_value,
                        self.shap_values[nth, :],
                        self.x.iloc[nth, :],
                        matplotlib=True)

    def feature_value_shap(self):
        shap.summary_plot(self.shap_values, self.x)

    def time_shap(self):
        shap.force_plot(base_value=self.explainer.expected_value,
                        shap_values=self.shap_values,
                        features=self.x)



    def depend_shap(self,depend_feature):
        depend_feature=str(depend_feature)
        shap.dependence_plot(depend_feature, self.shap_values, self.x)

    def mean_shap(self):
        shap.summary_plot(self.shap_values, self.x, plot_type="bar")

    def intera_shap(self):
        shap_interaction_values = self.explainer.shap_interaction_values(self.x)
        shap.summary_plot(shap_interaction_values, self.x)


