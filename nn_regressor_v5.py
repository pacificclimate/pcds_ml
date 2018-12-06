import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time
import loaders

def add_nan(record,slice_index):

    if slice_index.shape[0] != record.shape[0]:
        for z in slice_index:
            if z not in record.index:
                record.at[z,:] = float('NaN')
        record.sort_index(axis=0,inplace=True)

    return record
def drop_missing(data):
    ycounter = 0
    ncounter = 0
    for row in data.index:
        if True in np.isnan(data.loc[row,:].values):
            ncounter += 1
            data.drop(labels=row,inplace=True)
        else:
            ycounter += 1
    print('{} observations'.format(ycounter))
    print('{} missing'.format(ncounter))
    for row in data.index:
        if True in np.isnan(data.loc[row,:].values):
            print('Still nan')
    return data
def simple_plot(record,variable):
    plt.scatter(record[0].index,record[0][variable],c='k')
    plt.title(record[1]['station_name'])
    plt.show()
def take_second(elem):
    return(elem[1])
def get_jd(date):
    jd = loaders.ytd_length[date.month] + date.day
    return jd
def add_window(data,testing):
    #Add empty columns to hold window of previous days
    for num in range(window_size):
        data['p{}'.format(num)] = pd.Series(np.nan,index=data.index)

    #Propogate values diagonally downward to fill empty columns
    for row in data.index:
        cell = data.at[row,'original']
        if row == ti and np.isnan(cell):
            data.loc
        #Remove random subset of points for testing
        if testing:
            if not np.isnan(cell):
                if np.random.random() <= missing_fraction:
                    removed_dates.append(row)
                    removed_vals.append(cell)
                    data.at[row,'original'] = np.nan
        for offset in range(window_size):
            data.at[row+(1+offset)*d,'p{}'.format(offset)] = cell

    data.drop(index=data.index[data.index.shape[0]-window_size:],inplace=True) #Drop extra rows created at the end

    return data

#Initialize parameters
testing = False              #if testing, randomly removes some data to estimate preciction error
missing_fraction = 0.2      #Fraction of data to remove when testing
window_size = 10             #Number of previous days to consider
variable = 'min_temp'       #Climate variable to consider
coverage_req = 0.8          #Fraction of missing observations in the target station that must be covered by a valid neighbor station
corr_threshhold = 0.7       #Correlation value required between target measurements and neighbor measurements
max_neighbors = 10          #Maximum number of correlated stations that will be loaded


#Define target station
nid = '120C036'
network = 'EC'
start_time = time.time()
try:
    source = loaders.load_gapless(network,nid)
except FileNotFoundError:
    if not testing:
        raise SystemExit('No record')


d = pd.Timedelta('1 days')
removed_dates = []
removed_vals = []
ti = source[0].index[0]
tf = source[0].index[-1]
slice_index = pd.date_range(ti,tf,freq='D')
print('Target station: {}'.format(source[1]['station_name']))
print('Network: {}'.format(source[1]['network_name']))
print('Variable: {}'.format(variable))
print('Record length: {}'.format(tf-ti))

#Format data
data = source[0].loc[ti:tf,variable]
data = data.to_frame()
data = data.astype('float64')
#Check percentage missing
nan_count = np.count_nonzero(np.isnan(data.values))
if nan_count >= len(data)/2:
    print('Record is {}% missing. Proceed y/n?'.format((np.round(nan_count,5) / len(data))*100))
    choice = input('> ')
    if choice == 'n':
        raise SystemExit()

#Search for eligible neighbor stations
valid_neighbors,neighbor_ids = loaders.load_correlated_gapless(source[1]['network_name'],source[1]['native_id'],variable,corr_threshhold)
if len(valid_neighbors) > max_neighbors:
    valid_neighbors = sorted(valid_neighbors,key=take_second)[:max_neighbors]
    #valid_neighbors = valid_neighbors[:max_neighbors]

#Select neighbors with overlapping records and combine target and neighbors into a single array
load_time = time.time()
reasons = []
co = 0
fail_counter = 0 #Number of invalid neighbors
n_names = [] #Names of selected neighbor stations
fill_count = 0 #Number of values in neighbor stations filled in with mean
for i,neighbor_record in enumerate(valid_neighbors):
    time_slice = neighbor_record[0].loc[ti:tf,variable]
    time_slice = time_slice.to_frame()
    #fill_frame = pd.DataFrame(index=slice_index,columns=['flag'],data=)

    if time_slice.shape[0] / slice_index.size > coverage_req and False in np.isnan(time_slice[variable].values):
        n_normal = pd.read_pickle('../new_yearly_normals/' + neighbor_record[2] + '/' + neighbor_record[3] + '.pickle')
        n_nan_count = 0 #Counts days where neighbor and target are both NaN
        time_slice = add_nan(time_slice,slice_index)  #Add NaN values if station failed add_nan script

        #Replace NaN values in neighbor with normal for that day
        this_mean = time_slice[variable].mean(skipna=True)
        for row in time_slice.index:
            if np.isnan(time_slice.at[row,variable]):
                if np.isnan(data.at[row,variable]):
                    n_nan_count += 1
                try:
                    time_slice.at[row,variable] = n_normal[get_jd(row),variable]
                except KeyError:
                    time_slice.at[row,variable] = this_mean
                #time_slice.at[row,variable] = this_std * np.random.randn() + this_mean
                fill_count += 1
        if n_nan_count < 1000:
        #if n_nan_count / slice_index.shape[0] < (1-coverage_req):    #Confirms selected neighbor sufficiently covers target's gaps
            co += 1
            data['n{}'.format(co)] = pd.Series(data=time_slice[variable].values,index=slice_index)
            n_names.append((loaders.name_lookup((neighbor_record[2],neighbor_record[3])),neighbor_record[2]))
            reasons.append('Good')
        else:
            reasons.append('Doesnt cover gap')
            fail_counter += 1
    else:
        reasons.append('Record too short')
        fail_counter += 1


if len(valid_neighbors)-1 <= fail_counter:
    print()
    if len(neighbor_ids) == len(valid_neighbors):
        for i,x in enumerate(neighbor_ids):
            print('{}, {}   {}'.format(loaders.name_lookup(x),x[0],reasons[i]))
    else:
        print(reasons)
    raise SystemExit('Fewer than two valid neighbors')

print('{} neighbors used'.format(len(valid_neighbors)-fail_counter))
print(n_names)
print('{} values filled in neighbor records'.format(fill_count))
print(reasons)
#print('Nearby stations: {}'.format(n_names))
data.rename(columns={variable:'original'},inplace=True)

neighbor_time = time.time()

#Add window of previous days' observations to each datapoint
data = add_window(data,testing)

propagate_time = time.time()

#Slice  data
nanless_data = data.copy()
nanless_data = drop_missing(nanless_data)
true_mean = nanless_data['original'].mean(skipna=True)
true_var = nanless_data['original'].var(skipna=True)

xff = data.copy()
xtrain = nanless_data.loc[:,'n1':'p{}'.format(window_size-1)]
ytrain = nanless_data.loc[:,'original']
target_std = ytrain.std()


slice_time = time.time()

#Train the model
net = MLPRegressor(hidden_layer_sizes=(50,20),max_iter=500,alpha=0.0001)
#net = MLPRegressor()
model = net.fit(xtrain,ytrain)
fit_time = time.time()

xff['flag'] = 'M' #Data flagged as measured, predicted, or outlier

#Forward Prediction
total_sqerror = 0 #Stores cumulative squared error of prediction when testing on artificially missing data
avg = [] #Stores average prediction for each day
nan_dates = [] #Stores dates of values predicted by network
predictions = [] #Stores predicted values with index mapping to nan_dates
outlier_dates = []
outliers = []
for i,row in enumerate(xff.index):
    cell = xff.at[row,'original']
    input = xff.loc[row,'n1':'p{}'.format(window_size-1)].values.reshape(1,-1)
    try:
        next = float(model.predict(input))
    except (ValueError,TypeError):
        next = xff.at[row,'n1']
    if np.isnan(cell):
        xff.at[row,'flag'] = 'P'
        nan_dates.append(row)
        predictions.append(next)
        xff.at[row,'original'] = next

        for offset in range(window_size):   #Diagonally propogate values into previous day columns
            xff.at[row+(1+offset)*d,'p{}'.format(offset)] = next

        if testing:
            if row in removed_dates:
                sqerror = (next - removed_vals[removed_dates.index(row)])**2
                if np.isnan(sqerror):
                    print('found nan')
                else:
                    total_sqerror += sqerror
    #Check for outliers
    elif abs(next) + 2*target_std < abs(cell):
    #elif abs(cell) > 15 and abs((next - cell) / cell) > 0.6:
        nan_dates.append(row)
        predictions.append(next)
        outliers.append(cell)
        outlier_dates.append(row)
        xff.at[row,'flag'] = 'O'

xff.drop(index=xff.index[xff.index.shape[0]-window_size:],inplace=True) #Drop extra rows created at the end
ff_time = time.time()

#Display metrics
pred_mean = np.mean(predictions)
pred_var = np.var(predictions)
print('\n \t\tMean \tVariance')
print('Measured \t{}\t{}'.format(np.round(true_mean,2),np.round(true_var,2)))
print('Predicted\t{}\t{}'.format(np.round(pred_mean,2),np.round(pred_var,2)))


if testing:
    print('\nPrediction RMSE')
    pred_rmse = (total_sqerror/len(removed_dates)) ** 0.5
    print(pred_rmse)


print('\nTimes:')
print('Records loaded: {}s'.format(np.round(load_time-start_time,4)))
print('Neighbors formatted: {}s'.format(np.round(neighbor_time-load_time,4)))
print('Values propagated: {}s'.format(np.round(propagate_time-neighbor_time,4)))
print('Data sliced: {}s'.format(np.round(slice_time-propagate_time,4)))
print('Model trained: {}s'.format(np.round(fit_time-slice_time,4)))
print('Forward propagated {}s'.format(np.round(ff_time-fit_time,4)))
print('Total: {}s'.format(np.round(time.time()-start_time,4)))


prediction_frame = pd.DataFrame(index=nan_dates,data=predictions)
measured_plot = source[0].loc[:,variable]

#Plot filled record
plt.scatter(source[0].loc[ti:tf,variable].index,measured_plot,c='k') #Measured Data
plt.scatter(nan_dates,predictions, c='g') #Predicted by NN
plt.scatter(outlier_dates,outliers, c='r') #Detected outliers
plt.legend(['Measured','Predicted','Flagged outliers'])
plt.title('{}, {}, {}'.format(source[1]['station_name'],source[1]['network_name'],variable))
plt.show()

#Plot with neighbors
nti = ti
ntf = tf
rows = len(n_names) + 1
n_slice = xff.loc[nti:ntf]
pred_slice = prediction_frame.loc[nti:ntf]
plt.subplot(rows,1,1)
plt.scatter(data.loc[nti:ntf,'original'].index,data.loc[nti:ntf,'original'].values,c='k') #Measured Data
plt.scatter(pred_slice.index,pred_slice, c='g') #Predicted by NN
plt.scatter(outlier_dates,outliers, c='r') #Detected outliers
plt.legend(['Measured','Predicted','Flagged outliers'])
plt.title('{}, {}, {}'.format(source[1]['station_name'],source[1]['network_name'],variable))
for fignum,name in enumerate(n_names):
    plt.subplot(rows,1,fignum+2)
    plt.scatter(n_slice.index,n_slice['n{}'.format(fignum+1)].values) #Neighbor Data
    plt.title(name)
plt.show()
