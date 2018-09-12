import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans,Birch,DBSCAN
from sklearn.preprocessing import StandardScaler

def load_by_nid(network,nid):
    df = pd.read_pickle('D:/PCIC/station_data/' + network + '/' + nid + '.pickle')
    metadata = pd.read_csv('metadata/md_current.csv')
    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == nid:
                if metadata.at[row,'flag']=='good':
                    station_md = metadata.loc[row,:]
                elif metadata.at[row,'flag']=='NO DATA':
                    print('Record is blank')
        except AttributeError:
            pass
    record = [[df,station_md]]

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
    '''
    t0 = (ti[0],tf[0])
    t1 = (ti[1],tf[1])
    t2 = (ti[2],tf[2])
    ti = pd.Timestamp(t0[0],t1[0],t2[0])
    tf = pd.Timestamp(t0[1],t1[1],t2[1])
    '''
    labels = []

    for variable in variables:
        for x in records:
            time_slice = x[0].loc[x[0]['obs_time'] < tf]
            time_slice = time_slice.loc[time_slice['obs_time'] > ti]



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

def cluster_plot(record,variable,ti,tf):
    time_slice = record[0][0].loc[record[0][0]['obs_time']>ti]
    time_slice = time_slice.loc[time_slice['obs_time']<tf]
    data = time_slice.loc[:,variable].values
    data = np.array(list(zip(time_slice.index.values,data)))
    data = StandardScaler().fit_transform(data)

    #clusters = DBSCAN(eps=0.1).fit(data) #for fahrenheit issue
    clusters = DBSCAN(eps=0.7).fit(data) #for flatline
    plt.scatter(time_slice.loc[:,'obs_time'].values,time_slice.loc[:,variable].values,c=clusters.labels_)
    plt.show()

#source = load_by_coords(49.788064, -126.051257,5,30000)
#source = load_by_coords(49.42805,-123.56484,543,30000)
#source = load_by_name('elphin')
#source = load_by_nid('env-aqn', 'E208096')
variable = 'temp_mean'
variables = ['temperature']
ti = pd.Timestamp(2006,8,1)
tf = pd.Timestamp(2010,1,1)
#cluster_plot(load_by_nid('env-aqn', 'E208096'),'temp_mean',pd.Timestamp(2006,8,1),pd.Timestamp(2010,1,1))
cluster_plot(load_by_name('elphin'),'temperature',pd.Timestamp(2009,10,1),pd.Timestamp(2009,11,10))
#plot_records(source,variables,ti,tf)
