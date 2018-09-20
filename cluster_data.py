import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,Birch,DBSCAN
from sklearn.preprocessing import StandardScaler
import utm
import scipy.spatial.distance as dist

def load_by_nid(network,nid):
    df = pd.read_pickle('D:/PCIC/station_data/' + network + '/' + nid + '.pickle')
    df = df.set_index('obs_time')
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

def load_by_name_exact(name):
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
            if metadata.at[row,'station_name'].lower() == name:
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

def plot_records(records,variables, ti, tf,color='b'):
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
            #print(type(x))
            time_slice = x[0].loc[ti:tf]



            if not time_slice.empty and variable in time_slice.dtypes and time_slice[variable].dtype != (object or NoneType):
                nan_check = []
                for z in range(10):
                    index = np.random.choice(time_slice.index)
                    nan_check.append(np.isnan(time_slice.at[index,variable]))
                if False in nan_check:
                    plt.scatter(time_slice.index,time_slice.loc[:,variable].values,c=color)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
                else:
                    print('Column was nan')
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()

def cluster_dbs(records,variable,ti,tf):
    time_slice = records[0].loc[ti:tf]
    data = time_slice.loc[:,variable].values
    data[data==None] = np.nan
    data = data.astype('float64')
    ind = time_slice.index.values[~np.isnan(data)]
    data = data[~np.isnan(data)]
    data = np.array(list(zip(ind,data)))
    try:
        data = StandardScaler().fit_transform(data)
    except ValueError:
        return None
    clusters = DBSCAN(eps=0.5).fit(data) #for fahrenheit issue
    #clusters = DBSCAN(eps=0.7).fit(data) #for flatline

    return clusters

def cluster_kms(records,variable,ti,tf):
    time_slice = records[0].loc[ti:tf]
    data = time_slice.loc[:,variable].values
    data[data==None] = np.nan
    data = data.astype('float64')
    ind = time_slice.index.values[~np.isnan(data)]
    data = data[~np.isnan(data)]
    data = np.array(list(zip(ind,data)))

    data = StandardScaler().fit_transform(data)

    clusters = KMeans(n_clusters=4).fit(data)
    return clusters

def remove_outliers(record,clusters,variable,ti,tf):
    time_slice = record[0].loc[ti:tf]
    time_slice = time_slice.loc[~np.isnan(time_slice.loc[:,variable].values)]
    for ix, obs_time in enumerate(time_slice.index):
        if clusters.labels_[ix] == -1:
            time_slice.at[obs_time,variable] = None
    return [(time_slice,record[1])]


#sources = load_by_coords(49.42805,-123.56484,543,30000) #Huge list for Van area
#sources = [to_ec_standards(load_by_name('elphin')[0])]  #flatline
#sources = [to_ec_standards(load_by_nid('env-aqn', 'E208096')[0])]  #fahrenheit
#source = [to_ec_standards(load_by_name('othello')[0])] #wonky normal, bad example
#sources = [to_ec_standards(load_by_name('zz good hope lake')[0])] #only has data in the summer, cuts out random winter points
sources = [to_ec_standards(load_by_name_exact('hope')[0])] #One bad point
#source = load_by_nid('ec','110FAG9')
#source = [load_by_name('squamish')[6]]
variable = 'min_temp'
#variables = ['temperature','temp_mean']
ti = pd.Timestamp(1980,8,1)
tf = pd.Timestamp(2010,1,1)
#plot_records(source,[variable],ti,tf)
#plot_records(source,variables,ti,tf)
#for i,x in enumerate(source):
#    source[i] = to_ec_standards(x)
safe_source = []
for x in sources:
    if x[1]['network_name'] in ('EC','BCH','ENV-AQN','FLNRO-WMB') and variable in x[0].columns and x[0].loc[:,variable].any():
        safe_source.append(x)
for source in safe_source:
    clusters = cluster_kms(source,variable,ti,tf)
    time_slice = source[0].loc[ti:tf]
    data = time_slice.loc[:,variable].values
    data[data==None] = np.nan
    data = data.astype('float64')
    ind = time_slice.index.values[~np.isnan(data)]
    data = data[~np.isnan(data)]
    data = pd.DataFrame(data=data,index=ind,columns=[variable])
    if clusters == None:
        labels = []
    else:
        labels = clusters.labels_
    #cleaned = remove_outliers((data,source[1]),clusters,variable,ti,tf)
    plot_records([(data,source[1])],[variable],ti,tf,color=labels)
    #plot_records(cleaned,[variable],ti,tf,color=labels)
