import sys
sys.path.append('../')
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import plot_recurrence as plr
import pandas as pd
import DataLoad
# from ipdb import set_trace

def rec_plot(s, eps=0.001, steps=10):
    if eps==None: eps=0.1
    if steps==None: steps=10
    N = s.size
#     set_trace()
    S = np.repeat(s[None,:], N, axis=0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps
    return Z

def torch_ESN(parameters): 
    device = torch.device('cuda')
    dtype = torch.double
    #HIData Mackey_glass
#     set_trace()
    [X_data,Y_data]=DataLoad.FilesLoad('Mackey_glass')
    torch.set_default_dtype(dtype)
    X_data = torch.from_numpy(X_data).to(device)
    Y_data = torch.from_numpy(Y_data).to(device)
    N1=5000
#     set_trace()
    trX = X_data[:N1]
    trY = Y_data[:N1]
    tsX = X_data[N1:]
    tsY = Y_data[N1:]
#     set_trace()
    washout = [500]
    input_size = trX.shape[2]
    output_size = 1
    hiddensize = abs(int((parameters[0]+0.5)*500))
    numlayers=abs(int((parameters[1]+0.05)*10))
    w_ih_scale=abs(parameters[2])*1
      
    loss_fcn = torch.nn.MSELoss()
     
    start = time.time()

        # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)
#         set_trace()
    model = ESN(input_size, hidden_size=hiddensize, output_size=output_size, num_layers=numlayers,w_ih_scale=w_ih_scale)
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
#     set_trace()
    output, hidden = model(trX, washout)
    
#         print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

            # Test
#         set_trace()
    output, hidden = model(tsX, [0], hidden)
    #set_trace()
    if loss_fcn(output, tsY).item()<1e-10:
        output1=output.reshape(shape=(len(output.tolist()),)).tolist()
        output1=np.array(output1)
#             scaler = preprocessing.StandardScaler()
#             output1 = scaler.fit_transform(output1)
        output1=output1/np.mean(output1)
        rec = rec_plot(output1)
        plt.imshow(rec, cmap = plt.cm.gray)
        plt.savefig('../Results/RecurrencePlots'+str(numlayers)+'_'+str(hiddensize)+'.png',dpi=600)
        plt.show()

    print("Test error:", loss_fcn(output, tsY).item())
#         print("Ended in", time.time() - start, "seconds.")
#         preds=[]
#         testD=[]
#         画图
#         for i in range(500):
#             preds.append(output.tolist()[i][0])
#             testD.append(tsY[i][0])
#         plt.plot(preds, color='red')
#         plt.plot(testD,'--' )
#         plt.plot()
#         plt.show()

    return loss_fcn(output, tsY).item()
