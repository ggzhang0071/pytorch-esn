import TorchESN as TN
import os
import torch
import numpy as np
from numpy import random as rnd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import SaveDataCsv as SV
from pdb import set_trace

np.seterr(divide='ignore', invalid='ignore')
# %config InlineBackend.figure_format = 'retina'

def euclid_dist(x,y):
    temp = 0   
    for i,j in zip(x,y):
        temp += (i-j)**2
        final = np.sqrt(temp)
    return final

max_iters = 50
c1 = 2
c2 = 2
num_particles =30
g0 = 1
hidden_nodes = 15  
dim =3
w1=2;                 
wMax=0.9            
wMin=0.5              
current_fitness = np.zeros((num_particles,1))
gbest = np.zeros((1,dim))
gbest_score = float('inf')
OldBest=float('inf')

convergence = np.zeros(max_iters)
alpha = 20
epsilon = 1

class Particle:
    pass

#all particle initialized
particles = []
for i in range(num_particles):
    p = Particle()
    p.params =np.array([np.random.randint(400,700), np.random.randint(1,10), np.random.uniform(-1,1)])

    p.fitness = rnd.rand()
    p.velocity = 0.3*rnd.randn(dim)
    p.res_force = rnd.rand()
    p.acceleration = rnd.randn(dim)
    p.force = np.zeros(dim)
    p.id = i
    particles.append(p)

#training 
for i in range(max_iters):
    # gravitational constant
    g = g0*np.exp((-alpha*i)/max_iters)
    # calculate mse
    cf = 0    
    for p in particles:
        fitness = 0
        y_train = 0
        print(p.params)
        [fitness,hidden0,named_parameters] = TN.torch_ESN(p.params)
        hiddensize=int(p.params[0])
        numlayers=int(p.params[1])
#         fitness = fitness/X.shape[0]
        OldFitness=fitness
        current_fitness[cf] = fitness
        cf += 1
        if gbest_score > fitness and OldBest>fitness:
#             set_trace()
            hiddenState=np.array(hidden0.view(numlayers,hiddensize).tolist())
            rp = RecurrencePlot(dimension=7, time_delay=3,
                        threshold='percentage_points',
                        percentage=30)
            X_rp = rp.fit_transform(hiddenState)
            plt.figure(figsize=(6, 6))
            plt.imshow(X_rp[0], cmap='binary', origin='lower')
    #         plt.title('Recurrence Plot', fontsize=14)
            plt.savefig('../Results/RecurrencePlots'+str(numlayers)+'_'+str(hiddensize)+'_'+str(fitness)+'.png',dpi=600)
            plt.show()
            weightsName='reservoir.weight_hh'
            for name, param in named_parameters:
    #             print(name,param)
                if name.startswith(weightsName):
    #                 set_trace()
                    torch.save(param,'../Results/weights'+str(fitness)+'.pt')
    
            OldBest=gbest_score
            gbest_score = fitness
            gbest = p.params
    
    best_fit = min(current_fitness)
    worst_fit = max(current_fitness)
    
    for p in particles:
        p.mass = (current_fitness[particles.index(p)]-0.99*worst_fit)/(best_fit-worst_fit)
    
    for p in particles:
        p.mass = p.mass*5/sum([p.mass for p in particles])
    
    
    # gravitational force
    for p in particles:
        for x in particles[particles.index(p)+1:]:
            p.force = (g*(x.mass*p.mass)*(p.params - x.params))/(euclid_dist(p.params,x.params))
    
    # resultant force
    for p in particles:
        p.res_force = p.res_force+rnd.rand()*p.force
    
    # acceleration
    for p in particles:
        p.acc = p.res_force/p.mass
    
    w1 = wMin-(i*(wMax-wMin)/max_iters)
    
    # velocity
#     set_trace()
    for p in particles:
        p.velocity = w1*p.velocity+rnd.rand()*p.acceleration+rnd.rand()*(gbest - p.params)
    
    # position
    for p in particles:
        p.params = p.params + p.velocity
    convergence[i] = gbest_score
    
fig1 = plt.gcf()
plt.subplot(111)    
plt.plot(ConvergenceChanges)  
plt.ylabel('Error')
plt.draw()
fig1.savefig('../Results/ConvergenceChanges.png',dpi=600)  
    
sys.stdout.write('\rMPSOGSA is training ESN (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
sys.stdout.flush()
    # save results 
Path='../Results/'
FileName='BestParameters.csv'
newdata=[max_iters,num_particles,p.params,convergence]
PathFileName=os.path.join(Path,FileName)
SV.SaveDataCsv(PathFileName,newdata)
