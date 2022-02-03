
import pandas as pd

f="D:\\SM data\\FLUXNET"


def make_desc_select_sites(f):
    '''

    :param f: sites total path.
    :return: selected sites and their description. A new csv
    '''
    file_name_desc = f + "\\" + "desc_sites.csv"
    desc_data = pd.read_csv(file_name_desc)

    file_name_sele = f + "\\" + "select_sites.csv"
    sele_data = pd.read_csv(file_name_sele, header=0)

    new_desc_sele_data=pd.DataFrame(columns=['SITE_ID', 'SITE_NAME', 'LOCATION_LAT', 'LOCATION_LONG',
           'LOCATION_ELEV', 'IGBP', 'MAT', 'MAP','SITE_PATH'])
    k=0
    for i in sele_data.index:
        if sele_data.loc[i]["quality"] ==1:
            for j in desc_data.index:
                if desc_data.loc[j]["SITE_ID"] == sele_data.loc[i]["site_names"]:
                    new_desc_sele_data.loc[k]=(
                            desc_data.loc[j]["SITE_ID"],
                            desc_data.loc[j]['SITE_NAME'],
                            desc_data.loc[j]['LOCATION_LAT'],
                            desc_data.loc[j]['LOCATION_LONG'],
                            desc_data.loc[j]['LOCATION_ELEV'],
                            desc_data.loc[j]['IGBP'],
                            desc_data.loc[j]['MAT'],
                            desc_data.loc[j]['MAP'],
                            sele_data.loc[i]['site_paths'])

                    k=k+1
    return new_desc_sele_data

# if __name__ == '__main__':
#     nn=make_desc_select_sites(f)
#
#     nn.to_csv("new_desc_sele_data.csv")


