import random
import matplotlib.pyplot as plt
import time
import numpy as np

class MyPSO:
    def __init__ (self, fitfunc, opt_type, ndim, bounds, npop, epoch, c1=2.05, c2=2.05):
        self.opt_type=opt_type
        self.bounds=bounds
        self.npop=npop
        self.size=bounds
        self.ndim=ndim
        self.epoch=epoch
        self.pop_hist=[]
        self.w=0.9
        self.v0=0.1
        self.c1=c1
        self.c2=c2
        self.glob_best=0
        self.local_bests=[]

        self.lowb=[self.bounds[i][0] for i in range(len(self.bounds))]
        self.upb=[self.bounds[i][1] for i in range(len(self.bounds))]

        if self.opt_type=='max':
            self.fitfunc=fitfunc
        if self.opt_type=='min': 
            def fitwrap(*args,**kwargs):
                return -fitfunc(*args,**kwargs)
            self.fitfunc=fitwrap
        else:
            raise ValueError("Non-defined optimization type, use max or min")

    def gen_pop(self):
        #create current population list
        current_pop=[[0]*3 for i in range(self.npop)]
        #fill initial population
        for i in range(self.npop):
            x_ndim=[]
            v_ndim=[]
            for j in range(self.ndim):
                upb=float(self.upb[j])
                lowb=float(self.lowb[j])
                x=random.uniform(lowb,upb)
                v=self.v0
                x_ndim.append(x)
                v_ndim.append(v)
            current_pop[i][0]=x_ndim
            current_pop[i][1]=v_ndim
        return current_pop

    def eval_fitfunc(self,cpop):
        #applies fitness function to population positions
        for i in range(self.npop):
            cpop[i][2]=self.fitfunc(cpop[i][0])
        return cpop
    
    def init_bests(self,cpop):
        #init global best
        self.glob_best=cpop[0]
        for i in range(self.npop):
            if cpop[i][2]>self.glob_best[2]:
                self.glob_best=cpop[i]
        #init local bests at current_population[particle_number][4]
        for i in range(self.npop):
            self.local_bests.append(cpop[i])
        return cpop

    def get_nextpop(self,cpop):
        next_pop=[0]*len(range(self.npop))
        r1=random.random()
        r2=random.random()
        x_ndim=[]
        v_ndim=[]
        for i in range(self.npop):
            for j in range(self.ndim):
                x=cpop[i][0][j]+cpop[i][1][j]
                v=cpop[i][1][j]*self.w+self.c1*r1*(self.local_bests[i][0][j]-cpop[i][0][j])+self.c2*r2*(self.glob_best[0][j]-cpop[i][0][j])
                x_ndim.append(x)
                v_ndim.append(v)
            next_pop[i]=[x_ndim,v_ndim,0,0]
        self.pop_hist.append(cpop)
        return next_pop
    
    def update_bests(self,cpop):
        #update global best
        for i in range(self.npop):
            if cpop[i][2]>self.glob_best[2]:
                self.glob_best=cpop[i]
        #update local bests
        for i in range(self.npop):
            if cpop[i][2]>self.local_bests[i][2]:
                self.local_bests[i]=cpop[i]

    def evolute(self,epoch):
        current_pop=self.gen_pop()
        current_pop=self.eval_fitfunc(current_pop)
        current_pop=self.init_bests(current_pop)
        for i in range(epoch):
            current_pop=self.get_nextpop(current_pop)
            current_pop=self.eval_fitfunc(current_pop)
            self.update_bests(current_pop)
            self.w=self.w-((5/9)/epoch)
        self.pop_hist.append(current_pop)
        if self.opt_type=='min':
            self.glob_best[2]*=-1
        return self.glob_best[0],self.glob_best[2]

"""
# For 2D case
    def animate(self):
        for i in range(self.epoch):
            x_2d_list=[0]*self.npop
            y_2d_list=[0]*self.npop
            for j in range(self.npop):
                x_2d_list[j]=self.pop_hist[i][j][0][0]
                y_2d_list[j]=self.pop_hist[i][j][0][1]
            plt.scatter(x_2d_list,y_2d_list)
            plt.show()
"""

def xsquared(pos_vector):
    if pos_vector==None:
        pos_vector=[0]
    result=sum(x**2 for x in pos_vector)
    return result

dimensions=5
lower_bound=-10.0
upper_bound=10.0
npop=10
epoch=10
bounds=[[]*2 for i in range(dimensions)]
for i in range(dimensions):
    bounds[i]=[lower_bound,upper_bound]

pso=MyPSO(xsquared,'min',dimensions,bounds,npop,epoch)

x_best,y_best=pso.evolute(epoch)

#pso.animate()

print('Best location is:',x_best)
print('The result is:',y_best)

"""
pop_hist=[step_num][particle_num][attribute] : epoch times npop times 4-parameter attribute list. 
cpop=[particle_num][attribute]
attribute[[x],[v],y,[local best attributes]]
"""