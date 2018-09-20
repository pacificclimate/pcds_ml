import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utm
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans,Birch,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
import time
from os import listdir
import math

def to_ec_standards(record):
    fixed = ['ENV-AQN','FLNRO-WMB','EC_raw']
    if record[1]['network_name'] in fixed:
        tdata = record[0].copy()
        for col in tdata.columns:
            if 'temp' not in col or 'air_temperature_y' in col:
                tdata = tdata.drop(columns=[col])
        dfmaxt = tdata.resample('D').max()
        dfmint = tdata.resample('D').min()
        pdata = record[0].copy()
        for col in pdata.columns:
            if 'precip' not in col:
                pdata = pdata.drop(columns=[col])
        dfp = tdata.resample('D').sum()
        df = pd.concat([dfmaxt,dfmint],axis=1)
        df = pd.concat([df,dfp],axis=1)
        #df = pd.merge(df,dfp,on='obs_time')
        try:
            df.columns = ['max_temp','min_temp','one_day_precipitation']
        except ValueError:
            pass
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

    #Find search term in metadata file, record station's md and native id
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

    #Get record, reindex it and format for output
    out = []
    for i, net_id in enumerate(nets_ids):
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        df = df.set_index('obs_time')
        out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def load_by_coords(lat,long,elev,rad):
    target_utm = utm.from_latlon(lat,long)
    target_coords = [target_utm[0],target_utm[1],elev]
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0
    metadata = pd.read_csv('metadata/md_current.csv')

    for row in range(metadata.shape[0]):
        try:
            ut = utm.from_latlon(metadata.at[row,'lat'],metadata.at[row,'lon'])
            if ut[2] != target_utm[2] or ut[3] != target_utm[3]:
                continue
            coords = [ut[0],ut[1],metadata.at[row,'elev']]
            if dist.euclidean(coords,target_coords) <= rad:
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
                    #print(metadata.at[row,'station_name'])
                    #print(ut)
                elif metadata.at[row,'flag']=='NO DATA':
                    ndcounter += 1
                elif metadata.at[row,'flag']=='NO RECORD':
                    nrcounter+= 1
        except (AttributeError,ValueError):
            pass

    out = []
    for i, net_id in enumerate(nets_ids):
        df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
        df = df.set_index('obs_time')
        out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def plot_on_map(records,labels=None):
    xcoords = []
    ycoords = []
    nets = []
    for x in records:
        xcoords.append(x[1]['lon'])
        ycoords.append(x[1]['lat'])
        nets.append(x[1]['network_name'])

    plt.scatter(xcoords,ycoords,c=labels)
    #for i,x in enumerate(records):
    #        plt.annotate(x[1]['station_name'],(xcoords[i],ycoords[i]))
    plt.legend(labels)
    plt.show()

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
            time_slice = x[0].loc[ti:tf]
            #if x[1]['network_name'] == 'ENV-AQN':
            #    print(time_slice.dtypes)

            if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
                nan_check = []
                for z in range(10):
                    index = np.random.choice(time_slice.index)
                    nan_check.append(np.isnan(time_slice.at[index,variable]))
                if False in nan_check:
                    plt.scatter(time_slice.index,time_slice.loc[:,variable].values)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
                else:
                    print('Column was nan')
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()

def calculate_dist_matrix(records,variable):
    '''
    last = pd.Timestamp(1800,1,1)
    first = pd.Timestamp(2050,1,1)
    for x in records:
        if x[0].index[0] < first:
            first = x[0].index[0]
        if x[0].index[-1] > last:
            last = x[0].index[-1]

    length = (last - first).days
    obs_array = np.full((length+1,len(records)),None)
    for i,x in enumerate(records):
        obs_array[(x[0].index[0]-first).days:(x[0].index[0]-first).days + x[0].shape[0],i] = x[0].loc[:,variable].values
    for i in range(obs_array.shape[1]):
        if obs_array[:,i][obs_array[:,i]!=None].any():
            try:
                obs_array[:,i][obs_array[:,i]==None] = np.nanmean(obs_array[:,i][obs_array[:,i]!=None])
            except AttributeError:
                print(obs_array[:,i][obs_array[:,i]!=None])
                print((obs_array[:,i][obs_array[:,i]!=None])[0].dtype)
    #print(obs_array)
    dist_mat = pairwise.pairwise_distances(obs_array)
    '''

    dist_mat = np.full((len(records),len(records)),None)
    #print(dist_mat)
    for i,rec_i in enumerate(records):
        for j,rec_j in enumerate(records):
            if i==j:
                dist_mat[i][j] = 0
            elif variable not in rec_j[0].columns:
                dist_mat[i][j] == 300
            else:
                overlap_i = rec_i[0].loc[rec_j[0].index[0]:rec_j[0].index[-1],variable]
                if overlap_i.size == 0:
                    continue
                overlap_j = rec_j[0].loc[overlap_i.index[0]:overlap_i.index[-1],variable].values
                overlap_i = overlap_i.values

                if overlap_i.size == overlap_j.size and overlap_i.size > 0:
                    #print(overlap_i,overlap_j)
                    diffs = []
                    for ind,obs in enumerate(overlap_i):
                        try:
                            if np.isnan(obs) or np.isnan(overlap_j[ind]):
                                continue
                        except TypeError:
                            pass
                        try:
                            diff = obs - overlap_j[ind]
                            diffs.append(diff)
                        except TypeError:
                            pass
                    if diffs != []:
                        diffs = np.array(diffs)
                        avg = np.mean(diffs)
                        #print(avg)
                        if not np.isnan(avg):
                            dist_mat[i][j] = abs(avg)
                elif overlap_i.size > 0:
                    #print('Problem')
                    pass
    full = dist_mat[~(dist_mat==None)]
    avg = np.mean(full)
    dist_mat[dist_mat==None] = avg

    return dist_mat

def yearly_average(record,variable):
    year = pd.date_range(start='1/1/2018',periods=365,freq='D')
    df = pd.DataFrame(data=np.full((365,1),0.0),index=year,columns=[variable])

    for day in year:
        date_data = []
        for obs in record[0].index:
            if day.day == obs.day and day.month == obs.month:
                if record[0].at[obs,variable] != None and not np.isnan(record[0].at[obs,variable]):
                    date_data.append(record[0].at[obs,variable])
        if len(date_data) > 0:
            df.at[day,variable] = sum(date_data) / len(date_data)
    for column in range(df.shape[1]):
        for i in range(365):
            if abs(df.iat[i,column]) < 0.01:
                df.iat[i,column] = df.iat[i-1,column]

    return df


#source = load_by_coords(49.382662, -121.473297,40,40000)
source = load_by_name('hope')+load_by_name('vancouver')+load_by_name('kamloops')+load_by_name('george')
#source = load_by_coords(49.42805,-123.56484,543,30000) #Huge list for Van area
#source = load_by_name('hope')
variable = 'max_temp'
safe_source = []
for x in source:
    cleaned = to_ec_standards(x)
    #if not (cleaned[1]['network_name']== 'FLNRO-WMB' and cleaned[1]['resolution'] == 'daily'):
    if cleaned[1]['network_name'] in ('EC','BCH','ENV-AQN','FLNRO-WMB','EC_raw') and cleaned[0].loc[:,variable].any():
        safe_source.append(cleaned)
        #print(cleaned[1]['network_name'])
        #    print(x[1])
print(len(safe_source))
variables = ['max_temp']
ti = pd.Timestamp(1850,8,1)
tf = pd.Timestamp(2018,9,1)
#plot_records(source,['current_air_temperature1','air_temperature'],ti,tf)

#plot_records(safe_source,variables,ti,tf)
#dist_matrix = calculate_dist_matrix(safe_source,'max_temp')

#for x in safe_source:
#    plot_records([x],variables,ti,tf)

'''
print('Calculating yearly normals')
X = np.full((len(safe_source),1095),None)
for i, src in enumerate(safe_source):
    try:
        X[i][:365] = yearly_average(src,variable).T.values
        X[i][-730:-365] = yearly_average(src,'min_temp').T.values
        X[i][-365:] = yearly_average(src,'one_day_precipitation').T.values
    except KeyError:
        print(src[1])
'''
X = np.empty((len(safe_source),365*3))

for i,rec in enumerate(safe_source):
    X[i] = np.loadtxt('yearly_normals/' + rec[1]['network_name'] + '/' + rec[1]['native_id'] + '.txt',delimiter=',')
    #print(X[i,-365],X[i,-364])
    for j,cell in enumerate(X[i]):
        if cell == None or np.isnan(cell):
            X[i,j] = 0

#X = np.loadtxt('yearly_normals/hope_max_min_imp.txt',delimiter=',')
#X = np.loadtxt('yearly_normals/pg_van_hope_kam_max_min.txt',delimiter=',')

#print('Scaling data')
#X = StandardScaler().fit_transform(X)
print('Clustering stations')
#agg = AgglomerativeClustering(n_clusters=20,affinity='euclidean',linkage='ward')
#clusters = agg.fit(X)
#clusters = DBSCAN(eps=0.9,min_samples=2).fit(X)
clusters = KMeans(n_clusters=10).fit(X[:,:-365])
labels = clusters.labels_
print(labels)
print(len(labels))
plot_on_map(safe_source,labels=labels)


for val in range(10):
    labs = []
    for ind in range(X.shape[0]):
        if labels[ind] == val:
            plt.scatter(pd.date_range(start='1/1/2018',periods=365*2,freq='D'),X[ind,:-365])
            labs.append((safe_source[ind][1]['network_name'],safe_source[ind][1]['station_name']))
    plt.legend(labs)
    plt.show()
