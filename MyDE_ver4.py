import random
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

class MyDE:
    def __init__ (self, fitfunc, opt_type, ndim, bounds, epoch, npop=50, F=0.5, CR=0.3, seed=None):
        #F seems to be weight w for z=a+w(b-c)
        #Why did they mix?
        #They didn't make the i==j dimension certain to be selected. I will. 
        #Also random.random()<=0.3 is selected too.
        self.opt_type=opt_type
        self.bounds=bounds
        self.npop=npop
        self.size=bounds
        self.ndim=ndim
        self.epoch=epoch
        self.F=F
        self.CR=CR

        random.seed(seed) 
        np.random.seed(seed)

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
        self.pop_hist=[]
        self.fit_hist=[]
        self.bests=[]
        #create current population list
        current_pop=[[0]*2 for i in range(self.npop)]
        #fill initial population
        for i in range(self.npop):
            x_ndim=[]
            for j in range(self.ndim):
                upb=float(self.upb[j])
                lowb=float(self.lowb[j])
                x=random.uniform(lowb,upb)
                x_ndim.append(x)
            current_pop[i][0]=x_ndim
        return current_pop
    
    def enforce_bounds(self,x,dim):
        if x<self.bounds[dim][0]:
            x=self.bounds[dim][0]
        elif x>self.bounds[dim][1]:
            x=self.bounds[dim][1]
        return x
    
    def eval_fitfunc(self,cpop):
        #applies fitness function to population positions
        for i in range(self.npop):
            cpop[i][1]=self.fitfunc(cpop[i][0])
        return cpop
    
    def get_xprime(self,cpop):
        xprime=[[0]*2 for i in range(self.npop)]
        x_ndim=[0]*self.ndim
        a_list=[]
        for i in range(self.npop):
            #get a, b and c indices
            a_index=random.randint(0,self.npop-1)
            while a_index == i:
                a_index=random.randint(0,self.npop-1)
            b_index=random.randint(0,self.npop-1)
            while b_index==a_index or b_index==i:
                b_index=random.randint(0,self.npop-1)
            c_index=random.randint(0,self.npop-1)
            while c_index==a_index or c_index==b_index or c_index==i:
                c_index=random.randint(0,self.npop-1)
            #get z
            z_pos=[0]*self.ndim
            for j in range(self.ndim):
                z_pos[j]=(cpop[a_index][0][j]+self.F*(cpop[b_index][0][j]-cpop[c_index][0][j]))
                z_pos[j]=self.enforce_bounds(z_pos[j],j)
            #create xprime position
            for dim in range(self.ndim):
                prob_est=random.random()
                if prob_est<=self.CR:
                    x_ndim[dim]=z_pos[dim]
                else:
                    x_ndim[dim]=cpop[i][0][dim]
            xprime[i][0]=copy.deepcopy(x_ndim)
        return xprime
    
    def compare_x_xprime(self,cpop,xprime):
        next_pop=[[0]*2 for i in range(self.npop)]
        for i in range(self.npop):
            if xprime[i][1]>cpop[i][1]:
                next_pop[i]=copy.deepcopy(xprime[i])
            else:
                next_pop[i]=copy.deepcopy(cpop[i])
        return next_pop
    
    def get_best(self,cpop):
        best=copy.deepcopy(cpop[0])
        for i in range(self.npop):
            if cpop[i][1]>best[1]:
                best = copy.deepcopy(cpop[i])
        self.bests.append(best)
    
    def evolute(self,epoch):
        current_pop=self.gen_pop()
        current_pop=self.eval_fitfunc(current_pop)
        for i in range(epoch):
            xprime=self.get_xprime(current_pop)
            xprime=self.eval_fitfunc(xprime)
            current_pop=self.compare_x_xprime(current_pop,xprime)
            self.get_best(current_pop)
        if self.opt_type=='min':
            for i in range(self.epoch):
                self.bests[i][1]*=-1
        return self.bests