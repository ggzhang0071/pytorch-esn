import numpy as np
import matplotlib.pyplot as plt
import TorchESNWithRegularation as TN
# from ipdb import set_trace

lamba=np.linspace(0,1,11)
hiddensize =[0,0.01,0.03,0.05,0.07,0.09,0.1]
numLyers=np.linspace(0,1,11)
lossH=[]
for w in range(len(numLyers)):
    for p in range(len(hiddensize)):
        lossH.append([])
        for k in range(len(lamba)):
            tmp=[]
            for i in range(20):
                tmp.append(TN.torch_ESN([hiddensize[p],numLyers[w],0.001,lamba[k]]))
            lossH[p].append(np.mean(np.array(tmp)))

    np.save('../Results/RegularizationChangesHiddensizewithNumlayers'+str(numLyers[w]),lossH)
    fig = plt.gcf()
    plt.subplot(111)
    plt.style.use('ggplot')
    leg=[]
    for i in range(len(hiddensize)):
        plt.plot(lossH[i], lw=2)
    plt.xlabel('Regularization changes')
    plt.ylabel('Error')
    plt.draw()
    plt.legend(tuple(hiddensize))
    fig.savefig('../Results/RegularizationChangesHiddensizewithNumlayers'+str(numLyers[w])+'.png',dpi=600)  
    plt.show()
