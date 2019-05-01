import numpy as np
import matplotlib.pyplot as plt
import TorchESNWithRegularation as TN
from ipdb import set_trace

lamba=np.linspace(0,1,11)
hiddensize =[0,0.01,0.03,0.05,0.07,0.1]
numLyers=np.linspace(1,9,5)
loss=[]
for p in range(len(numLyers)):
    loss.append([])
    for k  in range(len(hiddensize)):
        loss[p].append([])
        for i in range(10):
            loss[p][k].append(TN.torch_ESN([hiddensize[k],0.3,0.001,lamba[i]]))

fig1 = plt.gcf()
plt.subplot(111)
plt.style.use('ggplot')
leg=[]
for p in range(len(numLyers)):
    for i in range(len(hiddensize)):
#         set_trace()
        leg.append(str(numLyers[p])+'_'+str(hiddensize[i]))
        plt.plot(loss[p][i], lw=2)
plt.xlabel('Regularization rate')
plt.ylabel('Error')
plt.draw()
plt.legend(tuple(leg))
fig1.savefig('../Results/RegularizationChangesAndHiddensize.png',dpi=600)  

plt.show()
