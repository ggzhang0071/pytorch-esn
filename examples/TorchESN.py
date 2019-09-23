import sys
sys.path.append('../')
import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from sklearn import preprocessing
# import matplotlib.pyplot as plt 
import pandas as pd
import DataLoad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

def torch_ESN(dataset,parameters,savepath): 
    device = torch.device('cuda')
    dtype = torch.double
    #HIData Mackey_glass
#     set_trace()
    [X_data,Y_data]=DataLoad.FilesLoad(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)
    torch.set_default_dtype(dtype)
    trX =torch.from_numpy(X_train).to(device)
    trY = torch.from_numpy(y_train).to(device)
    tsX =torch.from_numpy(X_test).to(device)
    tsY = torch.from_numpy(y_test).to(device)
    washout = [500]
    input_size = trX.size(2)
    output_size = 1
    hiddensize = abs(int(parameters[0]))
    numlayers=abs(int(parameters[1]))
    w_ih_scale=parameters[2]
    loss_fcn = torch.nn.MSELoss()
    start = time.time()

        # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)
    model = ESN(input_size, hidden_size=hiddensize, output_size=output_size, num_layers=numlayers,w_ih_scale=w_ih_scale)
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden0 = model(trX, washout)
    
#         print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

            # Test
    output, hidden = model(tsX, [0], hidden0)            
    print("Test error:", loss_fcn(output, tsY).item())

    print("Ended in", time.time() - start, "seconds.")
    preds=[]
    testD=[]
#         画图
    loss=loss_fcn(output, tsY).item()
    PlotLength=100
    if loss<3e-4:
        for i in range(tsY.size(0)):
            preds.append(output.tolist()[i][0])
            testD.append(tsY[i][0])
        plt.style.use('ggplot')
        x1=np.linspace(1,PlotLength,PlotLength)
        x2=np.linspace(PlotLength+1,2*PlotLength,PlotLength) 
        Y=torch.cat((trY[:PlotLength], tsY[:PlotLength]), 0).cpu().detach().numpy()[:,0,0]
        plt.plot(Y,lw=1.5,color='orangered')
        plt.plot(x1, output.cpu().detach().numpy()[:PlotLength][:,0][:,0],'--+',lw=1.5,color='magenta')
        plt.plot(x2,preds[:PlotLength],'--+',lw=1.5,color='green')
        plt.legend(('Original','Prediction (Training)','Prediction (Testing)'))
        plt.savefig(savepath+'Prediction_results/'+dataset+'_Prediction_results'+str(round(loss,5))+'.png',dpi=600)
    else:
        pass

    return loss,hidden0,model.named_parameters()
