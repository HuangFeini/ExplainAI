import pandas as pd
import datetime as dt



def time_add(data):
    '''
    Add time-relating features to input matrix.
    DAY: day sequence of entire observation
    DOY: day of year

    :parameter
    file: pd.Dataframe, data ,  from csv
    :returns
    data:pd.Dataframe

    '''
    DAY=[i for i in range(1,data.shape[0]+1)]
    DOYList=[]

    for i in data['TIMESTAMP']:
        time=str(i)
        timet=dt.datetime.strptime(time, "%Y%m%d")
        doy=timet.timetuple().tm_yday
        DOYList.append(doy)


    data['DAY']=DAY
    data['DOY']=DOYList
    data=data.drop(['TIMESTAMP'], axis=1)

    return data

def lag_add(data,sm_lag=7,p_lag=7):
    '''

    :param data: pd.Dataframe, input data
    :param sm_lag: int, lagged days of soil moisture
    :param p_lag: int, lagged days of precipitation
    :return:
    data: pd.Dataframe, input data with lagged variables
    '''
    if sm_lag-0>0:
        sm=data["SWC_F_MDS_1"].tolist()
        lag_col=["SM"+str(x) for x in range(1,sm_lag+1)]

        lag_frame=pd.DataFrame(columns=lag_col)
        i=0
        for col in lag_col:
            i=i+1
            lag=[int(-9999)]*i+sm[:-i]
            lag_frame[col]=lag
        data=pd.concat([data,lag_frame],axis=1)

    if p_lag-0>0:
        p = data["P_F"].tolist()
        lag_col = ["P" + str(x) for x in range(1, p_lag + 1)]

        lag_frame = pd.DataFrame(columns=lag_col)
        i = 0
        for col in lag_col:
            i = i + 1
            lag = [int(-9999)] * i + p[:-i]
            lag_frame[col] = lag
        data = pd.concat([data, lag_frame], axis=1)

        return data
    else:
        return data

