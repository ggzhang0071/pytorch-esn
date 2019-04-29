import configparser
import numpy as np
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

        
    if Datasets  =='HIData':
        datapath="../datasets/HIData/"
        dataname="input"
        dataextendname=".csv"
        datanameout="output"
        delimiter=','
        X_data=np.loadtxt(datapath+dataname+dataextendname, delimiter=delimiter, dtype=np.float64)
        Y_data= np.loadtxt(datapath+datanameout+dataextendname, delimiter=delimiter, dtype=np.float64)
        X_data = np.expand_dims(X_data, axis=1)
        Y_data = np.expand_dims(np.expand_dims(Y_data, axis=1),axis=1)
        
    return X_data,Y_data
        

