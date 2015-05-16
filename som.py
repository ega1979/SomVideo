# -*- coding: utf-8 -*-
# Copyright (C) 2014 Steven Shimizu All Rights Reserved.
# 
import math, matplotlib.pyplot as pl, matplotlib.animation as animation, numpy as np
import random, sys, datetime
RunStep = -1

class SomMap: # Self-organizing maps
    def __init__(self, size, count, rate, nck):
        self.Anime = None
        self.size = size
        self.size1 = None
        self.dim = len(size)
        self.count = count
        self.rate = rate
        self.map = None
        self.xs = None
        self.xsId = None
        self.id = None
        self.nck = nck
        self.nc0 = None
        self.colorGraph = False
        self.plotGraph = False
        self.plotSize = None

    def __call__(self,xs): return np.array([self.index(self.bmu(self.map, x)) for x in xs])
    def index(self,pos): 
        val = pos; index = []
        for d in xrange(self.dim-1,-1,-1):
            val,mod = divmod(val,self.size[d])
            index.insert(0,mod)
        return index
    def dist(self,v0,v1): return (sum([(v0[i]-v1[i])**2 for i in xrange(len(v0))]))**0.5
    def bmu(self,map,x):   # (Best Maching Unit)
        minD = None
        for i,e in enumerate(map):
            d = self.dist(x, e)
            if minD == None or d < minD: minD=d; minI = i
        return minI
    def trainBase(self,xs,xsId,id):
        self.xs = xs
        self.xsId = xsId
        self.id = id
        self.size1 = list(self.size)
        self.size1.append(len(xs[0]))
        elms = 1; minIndex = []; maxIndex = []
        for i in xrange(self.dim): 
            elms *= self.size[i]
            minIndex.append(0);maxIndex.append(self.size[i])
        self.map = np.array([[random.random() for i in xrange(len(xs[0]))] for i in xrange(elms)])
        self.nc0 = self.dist(minIndex,maxIndex)*self.nck

    def train(self,xs):
        self.Anime = False
        self.trainBase(xs,[],'')
        print 'LearnCount=',self.count,':',
        for t in xrange(self.count):
            print t, ; sys.stdout.flush()
            self.trainOne(t)
    def trainAnime(self,xs,xsId,id,colorGraph=False,plotGraph=False,plotSize=[1]):
        self.colorGraph = colorGraph
        self.plotGraph = plotGraph
        self.plotSize = plotSize
        self.Anime = True
        self.trainBase(xs,xsId,id)
        fig =  pl.figure()
        ani = animation.FuncAnimation(fig, self._update, frames = self.count, interval = 1,repeat=False) 
        pl.show()
        self.makeGraph(); pl.title(self.id); pl.savefig(id + '.png')

    def _update(self,i):
        self.makeGraph()
        pl.title(self.id + ' t:'+str(i)+' T:'+str(self.count))
        self.trainOne(i)
    def makeGraph(self):
        pl.clf()
        map = self.map.reshape(self.size1)
        mapped = self.__call__(self.xs)
        if self.colorGraph: pl.imshow(map,origin='lower',interpolation='none')
        else: 
            pl.xlim(0, self.size[1]);pl.ylim(0, self.size[0])
            if self.plotGraph: 
                dim = len(self.plotSize)
                if dim == 2:
                    meshSize = list(self.plotSize); meshSize.append(2); 
                    mapped2 = mapped.reshape(meshSize)
                    for i in xrange(self.plotSize[0]):
                        x,y = zip(*mapped2[i,:]); pl.plot(y, x)
                    for j in xrange(self.plotSize[1]):
                        x,y = zip(*mapped2[:,j]); pl.plot(y, x)
                else:
                    x,y = zip(*mapped); pl.plot(y, x)
        for j, m in enumerate(mapped):
            pl.text(m[1], m[0], self.xsId[j], ha='left', va='center',bbox=dict(facecolor='white', alpha=0.5, lw=0))
    def trainOne(self,t):
        global RunStep
        dec = (self.count - t) / float(self.count)
        a = self.rate * dec
        nc = self.nc0 *dec
        map = self.map
        for i,x in enumerate(self.xs):
            bmu = self.bmu(self.map, x)
            for j,e in enumerate(map):
                c = nc -  self.dist(self.index(bmu), self.index(j))
                if c > 0 : map[j] = [mv+a*(iv-mv) for iv, mv in zip(x, map[j])]
        if self.Anime and RunStep != -1 :
            if RunStep == 0 : RunStep = input('runStep>>>  ')
            RunStep -= 1
        
# --- data
colorNames = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']
colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])
animalNames = ['Hen','Duck','Hawk','Eagle','Fox','Dog','Wolf','Cat','Tiger','Lion','Horse','Cow','human']
animals = np.array(
#  Small Med Big 2legs 4legs Hair  Feathers Hunt Run Fly Swim 
 [[1.,   0., 0., 1.,   0.,   0.,   1.,      0.,  0., 0., 0.],#Hen
  [1.,   0., 0., 1.,   0.,   0.,   1.,      0.,  0., 1., 1.],#Duck
  [1.,   0., 0., 1.,   0.,   0.,   1.,      1.,  0., 1., 0.],#Hawk
  [0.,   1., 0., 1.,   0.,   0.,   1.,      1.,  0., 1., 0.],#Eagle
  [0.,   1., 0., 0.,   1.,   1.,   0.,      1.,  0., 0., 0.],#Fox
  [0.,   1., 0., 0.,   1.,   1.,   0.,      0.,  1., 0., 0.],#Dog
  [0.,   1., 0., 0.,   1.,   1.,   0.,      1.,  1., 0., 0.],#Wolf
  [1.,   0., 0., 0.,   1.,   1.,   0.,      1.,  0., 0., 0.],#Cat
  [0.,   0., 1., 0.,   1.,   1.,   0.,      1.,  0., 0., 0.],#Tiger
  [0.,   0., 1., 0.,   1.,   1.,   0.,      1.,  1., 0., 0.],#Lion
  [0.,   0., 1., 0.,   1.,   1.,   0.,      0.,  1., 0., 0.],#Horse
  [0.,   0., 1., 0.,   1.,   1.,   0.,      0.,  0., 0., 0.],#Cow
  [0.,   0., 1., 1.,   0.,   0.,   0.,      0.,  0., 0., 0.]])#human

def color1():
    size=(20,1); count=100; alpha= 0.1; nck = 0.5
    id = 'SOM Color '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(colors,colorNames,id,colorGraph=True)
def color2():  
    size=(20,30); count=50; alpha= 0.1; nck = 0.5
    id = 'SOM Color '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(colors,colorNames,id,colorGraph=True)

def animal1():
    size=(20,1); count=50; alpha= 0.1; nck = 1.0
    id = 'SOM Animal '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(animals,animalNames,id)
def animal2():
    size=(20,30); count=50; alpha= 0.1; nck = 1.0
    id = 'SOM Animal '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(animals,animalNames,id)
def animal3():
    size=(10,10,10); count=50; alpha= 0.1; nck = 1.0
    id = 'SOM Animal3 '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.train(animals)
    colors = 0.1*som(animals)
    print ''; print animalNames,colors 
    #
    size=(20,30); count=50; alpha= 0.1; nck = 0.5
    id = 'SOM Animal3 '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(colors,animalNames,id,colorGraph=True)

def one2two():
    dataSize = 30
    oneNames = [str(i) for i in xrange(dataSize)]
    ones = np.array([[i*1.0/dataSize] for i in xrange(dataSize)])
    size=(20,30); count=30; alpha= 0.1; nck = 0.5
    id = '1to2 '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(ones,oneNames,id,plotGraph=True)
def two2two():
    meshSize=(7,10)
    size=(20,30); count=50; alpha= 0.1; nck = 1.0
    twoNames = [str(i)+str(j) for i in xrange(meshSize[0]) for j in xrange(meshSize[1])]
    twos = np.array([[i*1.0/meshSize[0],j*1.0/meshSize[1]] for i in xrange(meshSize[0]) for j in xrange(meshSize[1])])
    id = '2to2 '+str((size,count,alpha,nck))
    som = SomMap(size, count, alpha, nck)
    som.trainAnime(twos,twoNames,id,plotGraph=True,plotSize=meshSize)

if __name__ == "__main__": 
    random.seed(7)
    argvs = sys.argv 
    if len(argvs) == 1: print 'Usage: python %s [color2|color1|animal2|animal1|animal3|1to2|2to2]' % argvs[0]
    if len(argvs) == 3 or len(argvs) == 2 :
        if len(argvs) == 3: RunStep = int(argvs[2])
        if argvs[1] == 'color1': color1()
        elif argvs[1] == 'color2': color2()
        elif argvs[1] == 'animal1': animal1()
        elif argvs[1] == 'animal2': animal2()
        elif argvs[1] == 'animal3': animal3()
        elif argvs[1] == '1to2': one2two()
        elif argvs[1] == '2to2': two2two()
        else: print 'keyword error'
