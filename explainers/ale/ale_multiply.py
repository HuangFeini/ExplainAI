from explainers.ale.ale_output import ale_output
from utils import _get_site_name,_get_site_IGBP,_get_site_feature_ale
import pandas as pd
import matplotlib.pyplot as plt
def _obtain_ale_res(f):
    for i in range(80,93):
        site_name=_get_site_name(f,i)
        ale_output(f,i)
        print(i,site_name)


def ale_IGBP_feauture_plot(f,IGBP,feature_name):
    data_file = f +"\\"+"new_desc_sele_data.csv"
    # get whole description as a dataframe
    desc_frame=pd.read_csv(data_file)

    site_paths_IGBP=[]
    site_index_IGBP=[]
    for i,f_IGBP in enumerate(desc_frame["IGBP"]):
        if f_IGBP==IGBP:
            site_paths_IGBP.append(desc_frame["SITE_PATH"][i])
            site_index_IGBP.append(desc_frame["INDEX"][i])

    print(site_paths_IGBP)
    print(site_index_IGBP)


    plt.figure()

    for index,ale in enumerate(site_index_IGBP):
        try:
            ale_data = _get_site_feature_ale(f, ale, feature_name)
            site_name=_get_site_name(f,ale)
            x=ale_data["quantiles"]
            y=ale_data["ALE"]
            plt.subplot(2,len(site_index_IGBP)/2,i)
            plt.plot(x,y)
            plt.text(0.5,0.5,str(site_name))
        except:
            print(index)
    plt.show()











    # site_IGBP=_get_site_IGBP(f,0)
    # print(site_IGBP)
    # ale_data=_get_site_feature_ale(f,0,"TS_F_MDS_1")
    # print(ale_data.shape)
    # plt.plot(ale_data["quantiles"],ale_data["ALE"])
    # plt.show()



if __name__ == '__main__':

    f = "D:\\SM data\\FLUXNET"
    # _obtain_ale_res(f)

    data_file = f +"\\"+"new_desc_sele_data.csv"
    # get whole description as a dataframe
    desc_frame=pd.read_csv(data_file)
    # get all IGBP
    site_all_IGBP=set(desc_frame["IGBP"])

    ale_IGBP_feauture_plot(f,"GRA","TS_F_MDS_1")





