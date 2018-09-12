import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans,Birch,DBSCAN
from sklearn.preprocessing import StandardScaler

def to_ec_standards(record):
    if record[1]['network_name'] == 'ENV-AQN':
        temp_var = 'temp_mean'
    if record[1]['network_name'] == 'EC-raw':
        temp_var = 'temp_mean'
    if record[1]['network_name'] == 'ENV-AQN' or record[1]['network_name'] == 'EC-raw':
        oldt = pd.DataFrame(data=record[0].loc[:,temp_var].values,index=record[0].loc[:,'obs_time'])
        dfmax = oldt.resample('D').max()
        dfmin = oldt.resample('D').min()
        #oldp = pd.DataFrame(data=record[0].loc[:,'precip_total'],index=record[0].loc[:,'obs_time'])
        #dfp = oldp.resample('D').sum()
        df = pd.merge(dfmax,dfmin,on='obs_time')
        #df = pd.merge(df,dfp,on='obs_time')
        df.columns = ['max_temp','min_temp']
        #df.columns = ['max_temp','min_temp','one-day_precipitation']
        return (df,record[1])
    else:
        return record

def load_by_name(name):
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')
    varnames = pd.read_csv('varnames_by_network.csv')

    for row in range(metadata.shape[0]):
        try:
            if name in metadata.at[row,'station_name'].lower():
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except AttributeError:
            pass

    for net_id in nets_ids:
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        if 'tm' in df.columns:
            df.rename(columns={'tm':'temp_mean'}, inplace=True)
        if 'current_air_temperature1' in df.columns:
            df.rename(columns={'current_air_temperature1':'temp_mean'}, inplace=True)
        if 'current_air_temperature2' in df.columns:
            df.rename(columns={'current_air_temperature2':'temp_mean'}, inplace=True)
        if 'air_temperature' in df.columns:
            df.rename(columns={'air_temperature':'temp_mean'}, inplace=True)
        if 'airtemp' in df.columns:
            df.rename(columns={'airtemp':'temp_mean'}, inplace=True)
        if 'tx' in df.columns:
            df.rename(columns={'tx':'max_temp'}, inplace=True)
        if 'maximum_air_temperature' in df.columns:
            df.rename(columns={'maximum_air_temperature':'max_temp'}, inplace=True)
        if 'tmax' in df.columns:
            df.rename(columns={'tmax':'max_temp'}, inplace=True)
        records.append(df)
    out = []
    for i in range(len(md)):
        out.append((records[i],md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def plot_records(records,variables, ti, tf):

    t0 = (ti[0],tf[0])
    t1 = (ti[1],tf[1])
    t2 = (ti[2],tf[2])
    ti = pd.Timestamp(t0[0],t1[0],t2[0])
    tf = pd.Timestamp(t0[1],t1[1],t2[1])

    labels = []

    for variable in variables:
        for x in records:
            time_slice = x[0].loc[x[0]['obs_time'] < tf]
            time_slice = time_slice.loc[time_slice['obs_time'] > ti]
            if x[1]['network_name'] == 'ENV-AQN':
                print(time_slice.dtypes)




            if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
                nan_check = []
                for z in range(10):
                    index = np.random.choice(time_slice.index)
                    nan_check.append(math.isnan(time_slice.at[index,variable]))
                if False in nan_check:
                    plt.scatter(time_slice.loc[:,'obs_time'].values,time_slice.loc[:,variable].values)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()


kam = load_by_name('kamloops')
for i in range(len(kam)):
    kam[i] = to_ec_standards(kam[i])
#for record in kam:
#    if record[1]['network_name'] == 'ENV-AQN':
#        print(record[1]['network_name'])
#        print(record[0].head(3))
variables = ['max_temp','min_temp']
ti = (2009,8,1)
tf = (2010,8,1)
plot_records(kam,variables,ti,tf)


'''
xcoords = []
ycoords = []
nets = []
for x in kam:
    xcoords.append(x[1]['lon'])
    ycoords.append(x[1]['lat'])
    nets.append(x[1]['network_name'])

    a = to_ec_standards(x)[0]
'''

#plt.scatter(xcoords,ycoords)
#plt.show()
