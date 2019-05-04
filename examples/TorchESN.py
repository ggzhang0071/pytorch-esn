import sys
sys.path.append('../')
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from pyts.image import RecurrencePlot
import pandas as pd
import DataLoad
# from ipdb import set_traceex

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
    hiddensize = abs(int(parameters[0]))
    numlayers=abs(int(parameters[1]))
    w_ih_scale=parameters[2]
      
    loss_fcn = torch.nn.MSELoss()
     
    start = time.time()

        # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)
#         set_trace()
    model = ESN(input_size, hidden_size=hiddensize, output_size=output_size, num_layers=numlayers,w_ih_scale=w_ih_scale)
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden0 = model(trX, washout)
    
#         print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

            # Test
    output, hidden = model(tsX, [0], hidden0)
    
    if loss_fcn(output, tsY).item()<1e-9:
        hiddenState=np.array(hidden0.view(numlayers,hiddensize).tolist())
        rp = RecurrencePlot(dimension=7, time_delay=3,
                    threshold='percentage_points',
                    percentage=30)
        X_rp = rp.fit_transform(hiddenState)
        plt.figure(figsize=(6, 6))
        plt.imshow(X_rp[0], cmap='binary', origin='lower')
#         plt.title('Recurrence Plot', fontsize=14)
        plt.savefig('../Results/RecurrencePlots'+str(numlayers)+'_'+str(hiddensize)+'_'+str(loss_fcn(output, tsY).item())+'.png',dpi=600)
        plt.show()
        weightsName='reservoir.weight_hh'
        for name, param in model.named_parameters():
#             print(name,param)
            if name.startswith(weightsName):
#                 set_trace()
                torch.save(param,'../Results/weights'+str(loss_fcn(output, tsY).item())+'.pt')
                
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
