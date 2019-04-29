def euclid_dist(x,y):
    temp = 0
    for i,j in zip(x,y):
        temp += (i-j)**2
        final = np.sqrt(temp)
    return final
    
from numpy import random as rnd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
import random
import matplotlib.pyplot as plt
from ipdb import set_trace
# %config InlineBackend.figure_format = 'retina'
max_iters = 4
c1 = 2
c2 = 2
num_particles = 2
g0 = 1
hidden_nodes = 15  
dim = 3
w=2;                 
wMax=0.9            
wMin=0.5              
current_fitness = np.zeros((num_particles,1))
gbest = np.zeros((1,dim))
gbest_score = float('inf')
OldBest=float('inf')
# input_nodes=1,
# hidden_nodes=500
# output_nodes=1
convergence = np.zeros(max_iters)
alpha = 20
epsilon = 1

class Particle:
    pass

#all particle initialized
particles = []
for i in range(num_particles):
    p = Particle()
    p.params =np.array([random.random() for i in range(dim)])
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
#         set_trace()
        fitness = TN.torch_ESN(p.params)
#         fitness = fitness/X.shape[0]
        OldFitness=fitness
        current_fitness[cf] = fitness
        cf += 1
        
        if gbest_score > fitness and OldBest>fitness:
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
    
    w = wMin-(i*(wMax-wMin)/max_iters)
    
    # velocity
    for p in particles:
        p.velocity = w*p.velocity+rnd.rand()*p.acceleration+rnd.rand()*(gbest - p.params)
    
    # position
    for p in particles:
        p.params = p.params + p.velocity
    
    convergence[i] = gbest_score
    sys.stdout.write('\rMPSOGSA is training ESN (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
    sys.stdout.flush()
    # save results
    Path='../Results/'
    FileName='BestParameters.csv'
    newdata=[max_iters,num_particles,p.params,convergence]
    PathFileName=os.path.join(Path,FileName)
