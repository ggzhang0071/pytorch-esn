import numpy as np
import matplotlib.pyplot as plt
import TorchESNWithRegularation1 as TN
# import TorchESNWithRegularation as TN

# from ipdb import set_trace

lamba=np.linspace(0.1,1,11)
hiddensize =np.linspace(0,1,6)
numLyers=[0. , 0.2, 0.4, 0.6, 0.8, 1. ]

def ESNRegularizationCompare(Monte_carlo,savePath):
    for w in range(len(numLyers)):
        lossH=[]
        for p in range(len(hiddensize)):
            lossH.append([])
            for k in range(len(lamba)):
                tmp=[]
                for i in range(Monte_carlo):
                    tmp.append(TN.torch_ESN([hiddensize[p],numLyers[w],1,lamba[k]]))
                lossH[p].append(np.mean(np.array(tmp)))

        np.save(savePath+'RegularizationChangesHiddensize'+str(numLyers[w]),lossH)
        fig1 = plt.gcf()
        plt.subplot(111)
        plt.style.use('ggplot')
        for i in range(len(hiddensize)):
            plt.plot(lossH[i], lw=2)
        plt.xlabel('Regularization rate')
        plt.ylabel('Error')
        plt.draw()
        plt.legend((10, 110, 210, 310, 410, 510))
        fig1.savefig(savePath+'RegularizationChangesHiddensize'+str(numLyers[w])+'.png',dpi=600)  
        plt.show()
