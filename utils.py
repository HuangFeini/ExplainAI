import pandas as pd


def _get_site_name(f,i):
    data_file = f +"\\"+"new_desc_sele_data.csv"
    site_name=pd.read_csv(data_file)["SITE_ID"][i]
    return site_name

def _get_site_DD_dataset_csv(f,i):
    '''获取经过全部数据集（经过全部的特征选择）'''
    site_path=_get_site_folder(f,i)
    data_path=site_path+"\\data_confirm.csv"
    data=pd.read_csv(data_path)
    return data


def _get_site_IGBP(f,i):
    data_file = f +"\\"+"new_desc_sele_data_origin.csv"
    site_IGBP=pd.read_csv(data_file)["IGBP"][i]
    return site_IGBP

def _get_site_feature_ale(f,i,feauture):
    site_path=_get_site_folder(f,i)
    prefix="ale_1_"
    if type(feauture) is str:
        ale_path=site_path+"\\"+prefix+feauture+".csv"
        ale_data=pd.read_csv(ale_path)
    return ale_data

def _get_version_res_folder(f,version,site_name=None,i=None):
    import os
    version_folder=f+"\\"+version
    if i:
        site_name=_get_site_name(f,i)
    elif site_name:
        site_name = site_name
    if os.path.exists(version_folder):
        site_version_res_folder=version_folder+"\\"+site_name
        if os.path.exists(site_version_res_folder):
            return site_version_res_folder
        else:
            os.mkdir(site_version_res_folder)
            return site_version_res_folder

def _get_site_folder(f,i=None,feature_name=None):
    data_file = f + "\\" + "new_desc_sele_data_origin.csv"
    data_content = pd.read_csv(data_file)
    print(feature_name)
    if type(i) is int:
        site_path=data_content["SITE_PATH"][i]
        return site_path
    elif type(feature_name) is str:
        site_path = data_content["SITE_PATH"][data_content["SITE_ID"]==feature_name].values[0]
        return site_path
    else:
        print("lack of index or feature_name.")








