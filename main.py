import io
import os
import numpy as np
#import tensorflow as tf
import baseSearch
import matplotlib.pyplot as plt
import time
from time import gmtime, strftime
from scipy.optimize import curve_fit
#=================================================================================
# data
path=os.getcwd()+"/data/artData.npy"
data=np.load(path)
x=data[:,0]
y=data[:,1]

#=================================================================================
# parameters
trainSetRatio=0.5
validationSetRatio=0.5
replica=1
runs=1
pathLength=2
branching=5


#=================================================================================
# problem solving

predict_vs_length=[]
xSpace=np.linspace(x[0],x[-1],500)

fringe=[]

ti=time.time()
for length in range(2,pathLength+1):
    predict=[]
    ti=time.time()
    branchNumSet=[]
    runTimes=[]
    for run in range(runs):
        #branching=2+run
        print(baseSearch.Color.CRED + 'run=%i/%i'%(run+1,runs) + baseSearch.Color.CEND)
        print
        pred=[]
        normalize=0
        for r in range(replica):
            print (baseSearch.Color.CRED +'      replica=%i/%i'%(r+1,replica)+ baseSearch.Color.CEND)
            problem=baseSearch.BaseSearch(x,y,trainSetRatio,validationSetRatio,replica,length,pathLength,branching,fringe)
            result=problem.train() #(error,[fit vector])
            fringe=result[2]
            w=(1/result[0])**3
            normalize+=w
            pred.append(np.dot(w,result[1]))
        predict.append(np.dot(1/normalize,np.sum(pred,0)))
    predict_vs_length.append(predict[0])
tf=time.time()
t=(tf-ti)
runTimes.append(t)
    #branchNumSet.append(branching)
    #print 'TE', predict[0]
#print 'ME', predict_vs_length[0]
#print 'lEngTh',len(predict_vs_length[0])

#plt.plot(x,y,'.')

#print' len PVL', len(predict_vs_length)
problem.plot(xSpace,predict_vs_length,np.mean(runTimes),np.arange(2,pathLength+1))


'''
plt.plot(branchNumSet,complexity,'k*')
plt.xlabel('branching number')
plt.ylabel('time complexity')
plt.savefig('/nfshome/ntatari/BaseSearch/July18/%s'%(strftime("%Y-%m-%d %H:%M:%S", gmtime())),dpi=300)
plt.show()
'''
#=================================================================================

def exp_decay_fit(data):
    n=np.arange(len(data))
    def func(x, a, b, c):
        return a*np.exp(-b*x) + c
    #print 'Here', n,data
    return curve_fit(func,n,data, maxfev=10000)[0][2]      

def convergence_with_path_length(predict_vs_length):
    c0=[]
    
    for x in range(len(predict_vs_length[0])):  
        data=[predict_vs_length[i][x] for i in range(len(predict_vs_length))]
        #print 'Here2',data
        c0.append(exp_decay_fit(data))
    return c0
   
c0=convergence_with_path_length(predict_vs_length)
#print 'c0',c0
#print 'x',xSpace
plt.plot(xSpace,c0)
plt.show()
#problem.plot(xSpace,c0,np.mean(runTimes),runs)

    
    
    
    
    
    
    
