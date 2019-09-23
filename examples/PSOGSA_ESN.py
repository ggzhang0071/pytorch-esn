## MPSOGSA
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
import argparse  
import SaveDataCsv as SV

def euclid_dist(x,y):
    temp = 0   
    for i,j in zip(x,y):
        temp += (i-j)**2
        final = np.sqrt(temp)
    return final
def PSOGSAESN(dataset,max_iters,num_particles,savepath):
    np.seterr(divide='ignore', invalid='ignore')
    # %config InlineBackend.figure_format = 'retina'
    c1 = 2
    c2 = 2
    g0 = 1
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
            [fitness,hidden0,named_parameters] = TN.torch_ESN(dataset,p.params,args.savepath)
            hiddensize=int(p.params[0])
            numlayers=int(p.params[1])
    #         fitness = fitness/X.shape[0]
            OldFitness=fitness
            current_fitness[cf] = fitness
            cf += 1
            if gbest_score > fitness and OldBest>fitness:
                """hiddenState=np.array(hidden0.view(numlayers,hiddensize).tolist())
                rp = RecurrencePlot()
                X_rp = rp.fit_transform(hiddenState)
                plt.figure(figsize=(6, 6))
                plt.imshow(X_rp[0], cmap='binary', origin='lower')
        #         plt.title('Recurrence Plot', fontsize=14)
                plt.savefig(savepath+'/RecurrencePlots/''RecurrencePlots'+str(numlayers)+'_'+str(hiddensize)+'_'+str(fitness)+'.png',dpi=600)
                plt.show()
                
                weightsName='reservoir.weight_hh'
                for name, param in named_parameters:
        #             print(name,param)
                    if name.startswith(weightsName):
        #                 set_trace()
                        torch.save(param,savepath+'weights'+str(fitness)+'.pt') """
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
        for p in particles:
            p.velocity = w1*p.velocity+rnd.rand()*p.acceleration+rnd.rand()*(gbest - p.params)

        # position
        for p in particles:
            p.params = p.params + p.velocity

        convergence[i] = gbest_score
#     set_trace()  
    plt.figure(figsize=(6, 6))
    plt.plot(convergence)  
    plt.xlabel('Convergence')
    plt.ylabel('Error')
    plt.draw()
    plt.savefig(savepath+'ConvergenceChanges.png',dpi=600)  

    sys.stdout.write('\rMPSOGSA is training ESN (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
    sys.stdout.flush()
        # save results 
    FileName='BestParameters.csv'
    newdata=[max_iters,num_particles,p.params,convergence]
    PathFileName=os.path.join(savepath,FileName)
    SV.SaveDataCsv(PathFileName,newdata)


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--dataset',default='Mackey_glass',
                    help='dataset to train')

parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

parser.add_argument('--max_iters', type=int,default=50,
                    help='')
parser.add_argument('--num_particles', type=int, default=30,
                    help='')
parser.add_argument('--savepath', type=str,required=False, default='../Results/',
                    help='Path to save results')

args = parser.parse_args()

if __name__ =="__main__":
    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
    PSOGSAESN(args.dataset,args.max_iters,args.num_particles,args.savepath)

