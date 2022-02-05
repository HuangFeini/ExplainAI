from explainers.ale.ale_output import ale_output,ale_plot_total
from explainers.ale.ale import ale_plot
from model.randomforest_gv import randomforest
import matplotlib.pyplot as plt
if __name__ == '__main__':
    best_model,data,r2=randomforest(re_feature_selection=False)

    d = data.drop("SWC", axis=1)
    ale_plot(best_model, train_set=d, features='TS', plot=False,bins=40, monte_carlo=False)

    ale_plot(best_model, train_set=d, features=tuple(['TS','DOY']), plot=False, bins=40, monte_carlo=False)
    plt.show()
    ale_plot_total(model=best_model,data=d)
    ale_output(model=best_model, data=d)


