import os
f="D:\\SM data\\FLUXNET\\SITEData"
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def _open_file(file_dir):
    '''
    :param file_dir:
    file_dir : input dir
    :return:
    site_names:
    site_paths:
    '''
    site_names=[]
    site_paths=[]
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        site_paths.append(root)
        site_names.append(root[32:38])
    return site_paths, site_names

def _read_raw_data(site_path,resolution="DD"):
    '''

    :param site_path:
    :param resolution:
    :return:
    '''
    if site_path:
        site_files = os.listdir(site_path)  #站点文件夹内所有文件
        for file in site_files:
            if resolution == "DD":
                if "FULLSET_DD" in file:
                    file_name=site_path+"\\"+ file
                    data = pd.read_csv(file_name)

                    return file_name,data


def _count_target(array, count_target):
    '''

    :param array:
    :param target:
    :return:
    '''
    arr = np.array(array)
    mask = (arr == count_target)
    arr_new = arr[mask]
    return arr_new.size

def get_FLX_quality(inputs, target="SWC_F_MDS_1", target_qc="SWC_F_MDS_1_QC", threshold=1000, resolution='DD'):
    """quality of FLUXNET2015 site data.
    Args:
        inputs ([type]):
            time series of site data.
        threshold ([type]):
            threshold of length of data. default set 1000 for daily data.
        resolution ([type]):
            time resolution of site data. default set as 'DD'. could be 'HH'.
        target([string]):
            a feature is prediction target. default set as "SWC_F_MDS_1".
    Returns:
        quality [type]:
            quality number of site data. 0 for bad site and 1 for good site.
    """


    length = len(inputs)

    # get threshold for specific time resolution
    if resolution == 'DD':
        threshold = threshold
    elif resolution == 'HH':
        threshold = threshold*48
    else:
        raise ValueError('Must daily or half-hour data.')

    if length < threshold:  # control length of inputs.
        quality = 0
    else:
        num_nan = np.sum(np.isnan(np.array(inputs)))
        if num_nan > 0.1*length:  # control length of NaN value.
            quality = 0
        else:
            if target in inputs.head():
                num_target_nan = np.sum(np.isnan(np.array(inputs[target])))
                # print("num_target_nan", num_target_nan)
                num_target_qc_1 = _count_target(array=inputs[target_qc], count_target=1)
                # print("num_target_qc_1", num_target_qc_1)
                if num_target_nan < 0.1*length:  # control length of NaN value of target feature.
                    if num_target_qc_1 >0.7*length: # control length of NaN value of target feature quality.
                        quality = 1
                    else:
                        quality = 0
                else:
                    quality = 0
            else:
                quality = 0


    return quality





def obtain_detected_sites(f):
    site_paths, site_names =_open_file(f)
    site_paths=site_paths[1:]
    site_names=site_names[1:]
    qc_list=[]

    for index, site_path in enumerate(site_paths):
        file_name,dat=_read_raw_data(site_path)

        if type(dat)=="NoneType":
            raise ValueError("The site folder is empty.")
            continue
        else:
            qc=get_FLX_quality(dat)
            qc_list.append(qc)
    return pd.DataFrame({"index":range(0,len(site_paths)),
                         "site_paths":site_paths,
                         "site_names":site_names,
                         "quality":qc_list})









# if __name__ == '__main__':
#
#     rr=obtain_detected_sites(f)
#
#     rr.to_csv("select_sites.csv")






