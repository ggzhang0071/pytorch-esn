import configparser
import numpy as np
from sklearn.preprocessing import normalize

def normal_std(x):
    normed = (x - x.mean(axis=0)) / x.std(axis=0)
    return normed

def Max_normalized(rawdata):
        #normalized by the maximum value of entire matrix.
    n, m = rawdata.shape;  
    normalize=rawdata.ndim
    data = np.zeros(rawdata.shape);
    if (normalize == 0):
        dat = rawdata
            
    if (normalize == 1):
        dat = rawdata / np.max(rawdata);
            
        #normlized by the maximum value of each row(sensor).
    if (normalize == 2):
        for i in range(m):
            scale = np.max(np.abs(rawdata[:,i]));
            data[:,i] = rawdata[:,i] / np.max(np.abs(rawdata[:,i]));
    return data

def  FilesLoad(Datasets):        
    #download DATA
    if Datasets  =='Mackey_glass':
        datapath="../datasets/"
        dataname="mg17"
        dataextendname=".csv"
        delimiter=','
        data = np.loadtxt(datapath+dataname+dataextendname, delimiter=delimiter, dtype=np.float64)
        X_data = np.expand_dims(data[:, [0]], axis=1)
        Y_data = np.expand_dims(data[:, [1]], axis=1)

        
    elif Datasets  =='HIData':
        datapath="../datasets/HIData/"
        dataname="input"
        dataextendname=".csv"
        datanameout="output"
        delimiter=','
        X_data=np.loadtxt(datapath+dataname+dataextendname, delimiter=delimiter, dtype=np.float64)
        Y_data= np.loadtxt(datapath+datanameout+dataextendname, delimiter=delimiter, dtype=np.float64)
        X_data = np.expand_dims(X_data, axis=1)
        Y_data = np.expand_dims(np.expand_dims(Y_data, axis=1),axis=1)
        
        
    elif Datasets  =='solar-energy':
        datapath='/git/data/TimeSeries/multivariate-time-series-data/solar-energy/'
        dataname="solar_AL"
        dataextendname=".txt"
        delimiter=','
        rawdata = np.loadtxt(open(datapath+dataname+dataextendname), delimiter=delimiter, dtype=np.float64)
        #data=normal_std(rawdata)
        data=normalize(rawdata)
        M = data.shape[1]
        X_data = data[:,:M-1]
        Y_data = data[:,M-1:]
        X_data = np.expand_dims(X_data, axis=1)
        Y_data = np.expand_dims(Y_data,axis=1)
        
        
    elif Datasets  =='traffic':
        datapath='/git/data/TimeSeries/multivariate-time-series-data/traffic/'
        dataname="traffic"
        dataextendname=".txt"
        delimiter=','
        data = np.loadtxt(datapath+dataname+dataextendname, delimiter=delimiter, dtype=np.float64)
        data=normal_std(data)
        X_data = np.expand_dims(data[:, [0]], axis=1)
        Y_data = np.expand_dims(data[:, [1]], axis=1)

        
    elif Datasets  =='traffic':
        datapath='/git/data/TimeSeries/multivariate-time-series-data/exchange_rate/'
        dataname="exchange_rate"
        dataextendname=".txt"
        delimiter=','
        data = np.loadtxt(open(datapath+dataname+dataextendname), delimiter=delimiter, dtype=np.float64)
        data=normal_std(data)
        X_data = np.expand_dims(data[:, [0]], axis=1)
        Y_data = np.expand_dims(data[:, [1]], axis=1)        
       
    elif Datasets  =='traffic':
        datapath='/git/data/TimeSeries/multivariate-time-series-data/electricity'
        dataname="electricity"
        dataextendname=".txt"
        delimiter=','
        data = np.loadtxt(open(datapath+dataname+dataextendname), delimiter=delimiter, dtype=np.float64)
        data=normal_std(data)
        X_data = np.expand_dims(data[:, [0]], axis=1)
        Y_data = np.expand_dims(data[:, [1]], axis=1)  
    else:
        print('data set is not existed.')
        
   
  
    return X_data,Y_data
        

