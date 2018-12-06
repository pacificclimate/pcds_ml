import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import utm
import scipy.spatial.distance as dist
import time

metadata = pd.read_csv('metadata/md_current.csv')
varnames = pd.read_csv('varnames_by_network.csv')
correlations = pd.read_pickle('correlation_matrix.pickle')
ytd_length = [np.nan,0,31,59,90,120,151,181,212,243,273,304,334]


def load_by_nid(network,nid,gapless=False):
    if gapless:
        df = pd.read_pickle('D:/PCIC/station_data_gapless/' + network + '/' + nid + '.pickle')
    else:
        df = pd.read_pickle('D:/PCIC/station_data/' + network + '/' + nid + '.pickle')
    df = df.set_index('obs_time')
    if network == 'ENV-AQN':
        nid = nid.lstrip('0')
    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == nid and metadata.at[row,'network_name'] == network:
                #if metadata.at[row,'flag']=='good':
                station_md = metadata.loc[row,:]
                break
                #elif metadata.at[row,'flag']=='NO DATA':
                #    print('Record is blank')
        except AttributeError:
            continue
    try:
        record = [df,station_md]
    except UnboundLocalError:
        #print('bad')
        #station_md = pd.DataFrame(data={'flag':'bad'},index=[1])
        record = [None,None]
    return record

def load_by_name(name,gapless=False):
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0


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
    if gapless:
        nets = ['EC','BCH','EC_raw','ENV-AQN','FLNRO-WMB']
        for i, net_id in enumerate(nets_ids):
            if net_id[0] in nets:
                df = pd.read_pickle('../station_data_gapless/'+net_id[0]+'/'+net_id[1]+'.pickle')
                #df = df.set_index('obs_time')
                out.append((df,md[i]))
    else:
        for i, net_id in enumerate(nets_ids):
            if net_id[0] == 'ENV-AQN' and net_id[1][0] not in ['E','M']:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/0'+net_id[1]+'.pickle')
            else:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
            df = df.set_index('obs_time')
            out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def load_by_name_exact(name,gapless=False):
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0

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
    if gapless:
        nets = ['EC','BCH','EC_raw','ENV-AQN','FLNRO-WMB']
        for i, net_id in enumerate(nets_ids):
            if net_id[0] in nets:
                df = pd.read_pickle('../station_data_gapless/'+net_id[0]+'/'+net_id[1]+'.pickle')
                out.append((df,md[i]))
    else:
        for i, net_id in enumerate(nets_ids):
            if net_id[0] == 'ENV-AQN' and net_id[1][0] not in ['E','M']:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/0'+net_id[1]+'.pickle')
            else:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
            df = df.set_index('obs_time')
            out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def load_by_coords(lat,long,elev,rad,gapless=False):
    target_utm = utm.from_latlon(lat,long)
    target_coords = [target_utm[0],target_utm[1],elev]
    records = []
    md = []
    nets_ids = []
    ndcounter = 0
    nrcounter = 0

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
    if gapless:
        nets = ['EC','BCH','EC_raw','ENV-AQN','FLNRO-WMB']
        for i, net_id in enumerate(nets_ids):
            if net_id[0] in nets:
                df = pd.read_pickle('../station_data_gapless/'+net_id[0]+'/'+net_id[1]+'.pickle')
                out.append((df,md[i]))
    else:
        for i, net_id in enumerate(nets_ids):
            if net_id[0] == 'ENV-AQN' and net_id[1][0] not in ['E','M']:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/0'+net_id[1]+'.pickle')
            else:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
            df = df.set_index('obs_time')
            out.append((df,md[i]))

    length = len(out)
    print('{} records loaded, {} were empty, {} missing'.format(length,ndcounter,nrcounter))
    return out

def load_nearby(record,rad,gapless=False):
    target_utm = utm.from_latlon(record[1]['lat'],record[1]['lon'])
    target_coords = [target_utm[0],target_utm[1],record[1]['elev']]
    records = []
    md = []
    nets_ids = []
    rad *= 1000

    for row in range(metadata.shape[0]):
        try:
            ut = utm.from_latlon(metadata.at[row,'lat'],metadata.at[row,'lon'])
            if ut[2] != target_utm[2] or ut[3] != target_utm[3]:
                continue
            coords = [ut[0],ut[1],metadata.at[row,'elev']]
            d = dist.euclidean(coords,target_coords,w=[1,1,100])
            if d <= rad and d != 0:
                if metadata.at[row,'flag']=='good':
                    nets_ids.append((metadata.at[row,'network_name'],metadata.at[row,'native_id'],metadata.at[row,'station_name']))
                    md.append(metadata.iloc[row])
        except (AttributeError,ValueError):
            pass

    out = []
    if gapless:
        nets = ['EC','BCH','EC_raw','ENV-AQN','FLNRO-WMB']
        for i, net_id in enumerate(nets_ids):
            if net_id[0] in nets:
                try:
                    df = pd.read_pickle('../station_data_gapless/'+net_id[0]+'/'+net_id[1]+'.pickle')
                    out.append((df,md[i]))
                except FileNotFoundError:
                    print('Gapless dataset missing station: {}'.format(md[i]['station_name']))
    else:
        for i, net_id in enumerate(nets_ids):
            if net_id[0] == 'ENV-AQN' and net_id[1][0] not in ['E','M']:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/0'+net_id[1]+'.pickle')
            else:
                df = pd.read_pickle('../station_data/'+net_id[0]+'/'+net_id[1]+'.pickle')
            df = df.set_index('obs_time')
            out.append((df,md[i]))

    length = len(out)
    print('{} records loaded'.format(length))
    return out

def simple_plot(df,variable):
    plt.scatter(df.index,df[variable],c='k')
    plt.show()

def load_ds(network,nid):
    df = pd.read_pickle('../deseasonalized_data/' + network + '/' + nid + '.pickle')

    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == nid and metadata.at[row,'network_name'] == network:
                station_md = metadata.loc[row,:]
                break
        except AttributeError:
            continue
    try:
        record = [df,station_md]
    except UnboundLocalError:
        #print('bad')
        station_md = pd.DataFrame(data={'flag':'bad'},index=[1])
        record = [df,station_md]
        print('!!!!!!!!!\n!!!!!!!!!!')
    return record

def load_gapless(network,nid):
    df = pd.read_pickle('../station_data_gapless/' + network + '/' + nid + '.pickle')

    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == nid and metadata.at[row,'network_name'] == network:
                station_md = metadata.loc[row,:]
                break
        except AttributeError:
            continue
    try:
        record = [df,station_md]
    except UnboundLocalError:
        #print('bad')
        station_md = pd.DataFrame(data={'flag':'bad'},index=[1])
        record = [df,station_md]
        print('!!!!!!!!!\n!!!!!!!!!!')
    return record

def load_correlated(network,nid,variable,corr_threshhold):
    corrs = correlations.loc[:,(network,nid,variable)]
    nearby = []
    for net,id in corrs.index:
        score = corrs[(net,id)]
        if score > corr_threshhold:
            nearby.append((net,id,score))
    records = []
    for n_net,n_nid,score in nearby:
        df = pd.read_pickle('D:/PCIC/deseasonalized_data/' + n_net + '/' + n_nid + '.pickle')
        records.append((df,score,n_net,n_nid))
    return records,nearby

def load_correlated_gapless(network,nid,variable,corr_threshhold):
    corrs = correlations.loc[:,(network,nid,variable)]
    nearby = []
    for net,id in corrs.index:
        score = corrs[(net,id)]
        if score > corr_threshhold:
            nearby.append((net,id,score))
    records = []
    for n_net,n_nid,score in nearby:
        df = pd.read_pickle('D:/PCIC/station_data_gapless/' + n_net + '/' + n_nid + '.pickle')
        records.append((df,score,n_net,n_nid))
    return records,nearby

def name_lookup(id):
    station_name = None
    for row in range(metadata.shape[0]):
        try:
            if metadata.at[row,'native_id'] == id[1] and metadata.at[row,'network_name'] == id[0]:
                station_name = metadata.at[row,'station_name']
                break
        except AttributeError:
            continue
    return station_name

def id_lookup(name):
    ids = (None,None)
    for row in range(metadata.shape[0]):
        try:
            if str(metadata.at[row,'station_name']) == name:
                ids = (metadata.at[row,'network_name'],metadata.at[row,'native_id'])
                break
        except AttributeError:
            continue
    return ids

def to_ec_standards(record):
    fixed = ['ENV-AQN','FLNRO-WMB','EC_raw']

    if type(record[1]) == type(None):
        return (None,None)

    if record[1]['flag'] in ['NO RECORD','NO DATA']:
        return (None,None)

    if record[1]['network_name'] in fixed:
        if 'max_temp' not in record[0].columns:
            tdata = record[0].loc[:,record[0].columns.str.contains('temp')].copy()
            tdata.drop(columns=tdata.columns[tdata.columns.str.contains('yesterday')],inplace=True)
            dfmaxt = tdata.resample('D').max()
            dfmint = tdata.resample('D').min()
        else:
            dfmaxt = record[0]['max_temp'].resample('D').max()
            dfmint = record[0]['min_temp'].resample('D').min()
        df = pd.concat([dfmaxt,dfmint],axis=1)

        pdata = record[0].loc[:,record[0].columns.str.contains('precip')].copy()
        if not pdata.empty:
            dfp = pdata.resample('D').sum()
            df = pd.concat([df,dfp],axis=1)
        try:
            df.columns = ['max_temp','min_temp','one_day_precipitation']
        except ValueError:
            df = df.iloc[:,:3]
        #print(df.columns)
        df = df.astype('float64')
        return (df,record[1])
    elif record[1]['network_name'] in ['EC','BCH']:
        df = record[0].loc[:,['max_temp','min_temp','one_day_precipitation']]
        df = df.astype('float64')
        return (df,record[1])
    else:
        #print(df.columns)
        return record

def plot_records(records,variables, ti, tf):

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
                    plt.scatter(time_slice.index,time_slice.loc[:,variable].values)
                    labels.append((x[1]['station_name'],x[1]['network_name'],variable))
                else:
                    print('Column was nan')
    plt.ylabel(variables)
    plt.legend(labels)
    plt.show()

if __name__ == '__main__':
    #julian_dict = pd.read_pickle('julian_dict.pickle')
    #a = load_gapless('FLNRO-WMB','111')
    #simple_plot(a[0],'max_temp')
    print(id_lookup('zz view (FD7)'))
