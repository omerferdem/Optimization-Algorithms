import random
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

class MyPSO:
    def __init__ (self, fitfunc, opt_type, ndim, bounds, npop, epoch, c1=2.05, c2=2.1, seed=None):
        self.opt_type=opt_type
        self.bounds=bounds
        self.npop=npop
        self.size=bounds
        self.ndim=ndim
        self.epoch=epoch
        self.v0=0.1
        self.c1=c1
        self.c2=c2
        self.r1_list=[]
        self.r2_list=[]

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
        self.local_bests=[]
        self.glob_best=[]
        self.w0=0.9
        self.w=self.w0
        self.wmin=0.4
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
                v=self.v0*x #*random.uniform(-1,1)
                x_ndim.append(x)
                v_ndim.append(v)
            current_pop[i][0]=x_ndim
            current_pop[i][1]=v_ndim
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
            cpop[i][2]=self.fitfunc(cpop[i][0])
        return cpop
    
    def init_bests(self,cpop):
        #init global best
        self.glob_best=copy.deepcopy(cpop[0])
        for i in range(self.npop):
            if cpop[i][2]>self.glob_best[2]:
                self.glob_best=copy.deepcopy(cpop[i])
        if self.opt_type=='min':
            self.fit_hist.append(self.glob_best[2]*(-1))
        else:
            self.fit_hist.append(self.glob_best[2])
        #init local bests at current_population[particle_number][4]
        for i in range(self.npop):
            self.local_bests.append(cpop[i])
        return cpop

    def get_nextpop(self,cpop):
        next_pop=[0]*len(range(self.npop))
        x_ndim=[]
        v_ndim=[]
        for i in range(self.npop):
            for j in range(self.ndim):
                r1=random.random()
                r2=random.random()
                self.r1_list.append(r1)
                self.r2_list.append(r2)
                
                
                cognitive_speed=self.c1*r1*(self.local_bests[i][0][j]-cpop[i][0][j])
                social_speed=self.c2*r2*(self.glob_best[0][j]-cpop[i][0][j])
                v=self.w*cpop[i][1][j] + cognitive_speed + social_speed
                x=cpop[i][0][j]+v
                x=self.enforce_bounds(x,j)
                x_ndim.append(x)
                v_ndim.append(v)
            next_pop[i]=[x_ndim,v_ndim,0]
            x_ndim=[]
            v_ndim=[]
        self.pop_hist.append(cpop)
        return next_pop
    
    def update_bests(self,cpop):
        #update local bests
        for i in range(self.npop):
            if cpop[i][2]>self.local_bests[i][2]:
                self.local_bests[i]=copy.deepcopy(cpop[i])
        #update global best
        for i in range(self.npop):
            if cpop[i][2]>self.glob_best[2]:
                self.glob_best=copy.deepcopy(cpop[i])
        if self.opt_type=='min':
            self.fit_hist.append(self.glob_best[2]*(-1))
        else:
            self.fit_hist.append(self.glob_best[2])
    
    def evolute(self,epoch):
        current_pop=self.gen_pop()
        current_pop=self.eval_fitfunc(current_pop)
        current_pop=self.init_bests(current_pop)
        for i in range(epoch):
            current_pop=self.get_nextpop(current_pop)
            current_pop=self.eval_fitfunc(current_pop)
            self.update_bests(current_pop)
            self.w=self.w0-(self.w0-self.wmin)*(i+1)/epoch
            #print('timew w=',self.w)
        self.pop_hist.append(current_pop)
        if self.opt_type=='min':
            self.glob_best[2]*=-1
        return self.glob_best[0],self.glob_best[2],self.fit_hist,self.pop_hist,self.r1_list,self.r2_list

    # For 2D case
    def animate(self):
        sct_legend=[]
        for i in range(10,self.epoch):
            x_2d_list=[0]*self.npop
            y_2d_list=[0]*self.npop
            for j in range(self.npop):
                x_2d_list[j]=self.pop_hist[i][j][0][0]
                y_2d_list[j]=self.pop_hist[i][j][0][1]
            im=plt.scatter(x_2d_list,y_2d_list,s=8)
            current_legend=['Step '+str(i+1)]
            sct_legend.append(current_legend)
        plt.legend(sct_legend)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        img=plt.show()

"""
pop_hist=[step_num][particle_num][attribute] : epoch times npop times 4-parameter attribute list. 
cpop=[particle_num][attribute] : current population, which is being generated or updated
attribute[[x],[v],y] : all the properties a particle needs to move, except global best
self.local_bests[i]: For i'th particle, the best position, speed at there, y value. The list holds the local best position.
"""