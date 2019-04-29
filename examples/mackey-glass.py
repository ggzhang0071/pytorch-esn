#all particle initialized
particles = []
for i in range(num_particles):
    p = Particle()
    p.params = np.array([rnd.rand() for i in range(dim)])
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
        weights1 = p.params[:60].reshape((4,15))
        weights2 = p.params[60:105].reshape((15,3))
        biases1 = p.params[105:120]
        biases2 = p.params[120:123]
        
        fitness = 0
        y_train = 0
        set_trace()
        for t in X:  #ith particle output compare with target , find fitness
            out = fnn(weights1, weights2, biases1, biases2, t[0], t[1], t[2], t[3])
            if Y[y_train] == -1:
                fitness = fitness + (1-out[0])**2 + (0-out[1])**2 + (0-out[2])**2
            elif Y[y_train] == 0:
                fitness = fitness + (0-out[0])**2 + (1-out[1])**2 + (0-out[2])**2
            elif Y[y_train] == 1:
                fitness = fitness + (0-out[0])**2 + (0-out[1])**2 + (1-out[2])**2
            y_train += 1
        
        fitness = fitness/X.shape[0]
        current_fitness[cf] = fitness
        cf += 1
        
        if gbest_score > fitness:
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
    sys.stdout.write('\rPSOGSA is training  (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
    sys.stdout.flush()
