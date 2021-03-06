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
#from ipdb import set_trace

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
    hiddensize = abs(int((parameters[0]+0.02)*500))
    numlayers=abs(int((parameters[1]+0.1)*20))
    w_ih_scale=abs(parameters[2])*1
      
    loss_fcn = torch.nn.MSELoss()
    lr = 1e-4
    weight_decay = parameters[3]# for torch.optim.SGD
    start = time.time()
    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)
    #         set_trace()
    model = ESN(input_size, hidden_size=hiddensize, output_size=output_size, num_layers=numlayers,w_ih_scale=w_ih_scale)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model(trX, washout, None, trY_flat)
    model.fit()
#     set_trace()
    output, hidden = model(trX, washout)
    output, hidden = model(tsX, [0], hidden)
    for t in range(100): 
        loss=loss_fcn(output, tsY)
        optimizer.zero_grad()
        loss.backward(retain_graph=True )

        optimizer.step()    
#     print("Training error:", loss_fcn(output, trY[washout[0]:]).item())
                # Test
    print("Test error:", loss.item())
    return loss.item()
