# Base_Search.py
import sys
import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import random
import Queue
import heapq
import copy
import time
from time import gmtime, strftime


class BaseSearch:

    def __init__(self,xData,yData,trainSetRatio,validationSetRatio,replica,length,pathLength,branching,fringe):

        '''        
        x=data[:,0]
        y=data[:,1]
        dx=tf.data.Dataset.from_tensor_slices(x)
        dy=tf.data.Dataset.from_tensor_slices(y)
        dataset=tf.data.Dataset.zip((dx,dy)).shuffle(30).repeat().batch(2)
        iterator=dataset.make_one_shot_iterator()
        next_element=iterator.get_next()

        with tf.Session() as sess:
            for i in range(40):
                print sess.run(next_element)
        '''
        
        self.replica=replica
        self.trainSetLen=int(trainSetRatio*len(xData))
        self.validationSetLen=int(validationSetRatio*(len(xData)-self.trainSetLen))
        self.xTrain=xData[:self.trainSetLen]
        self.yTrain=yData[:self.trainSetLen]
        self.xValid=xData[self.trainSetLen:self.trainSetLen+self.validationSetLen]
        self.yValid=yData[self.trainSetLen:self.trainSetLen+self.validationSetLen]
        self.xTest=xData[self.trainSetLen+self.validationSetLen:]
        self.yTest=yData[self.trainSetLen+self.validationSetLen:]
        self.xData=xData
        self.yData=yData
        self.trainSetTrust=(trainSetRatio/(validationSetRatio*(1-trainSetRatio)))/(1+(trainSetRatio/(validationSetRatio*(1-trainSetRatio))))
        self.dataSD=np.sqrt(np.var(self.yTrain))
        self.length=length
        self.pathLength=pathLength
        self.branching=branching
        self.fringe=fringe
        

        
    def treeSearch(self):
        root =[[['one']]]
        errors=[]
        appendings=[]
        y=np.append(self.yTrain,self.yValid)
        initDeviate=[(np.mean(self.yTrain)-y[i]) for i in range(len(y))]
        initError=np.linalg.norm(initDeviate)
        heapq.heappush(self.fringe,(100,[root,initError,0,initDeviate,[]]))
        loop=0
        errorMin=100
        actionCost=0
        xSpace=np.linspace(self.xData[0],self.xData[-1],500)
        
        
        while self.fringe:
            loop+=1
            current=heapq.heappop(self.fringe)
            error=current[0]
            parentFunction=current[1][0]
            depth=len(parentFunction)
            print '\n\n\n\n\n'
            print ('expansion:%i, length:(%i/%i)/(%i/%i) ' %(loop,len(parentFunction),self.length,self.pathLength,self.pathLength), [[parentFunction[i][j][0] for j in range(len(parentFunction[i]))] for i in range(len(parentFunction))])
            print

            

#            if error<errorMin: #as we go in more loops this part saves the node with the least error observed
#                bestNode=current
#                errorMin=error

            if (len(current[1][0])==self.length):
                bestNode=current[1]
                parentFunction=bestNode[0]
                error=bestNode[1]+bestNode[2]
                coefs=bestNode[4]
                kerMat=[[kerVal(x,base) for base in parentFunction] for x in xSpace]
                predict=self.prediction(kerMat,coefs,xSpace)
                print 'baseFunctions',parentFunction #[[parentFunction[i][j][0] for j in range(len(parentFunction[i]))] for i in range(len(parentFunction))]
                return (error,predict,self.fringe)
                
                

            appendings=randomBase(self.branching)
            hParent=current[1][1]   
            gParent=current[1][2]
            vParent=current[1][3]
            successors=self.getSuccessors(parentFunction,appendings,hParent,gParent,vParent,loop)
            for newNode in successors:
                h=newNode[1]
                g=newNode[2]
                f=h+g
                heapq.heappush(self.fringe,(f,newNode))
            
            #actionCostNode=heapq.heappop(fringe) #next node with a bit higher error than the current node
            #actionCost=10*(actionCostNode[0]-error) #difference node error with its next larger error node same depth
            #heapq.heappush(fringe,(actionCostNode[0],actionCostNode[1]))



    def getSuccessors(self,parentFunction,appendings,hParent,gParent,vParent,loop):
        successors=[]
        sNum=0
        self.kerMat=[[kerVal(x,base) for base in parentFunction] for x in self.xData]
        for appending in [appendings[i] for i in range(len(appendings))]:
                sNum+=1
                print (Color.CBLUE+'child(%i/%i):'%(sNum,len(appendings))+Color.CEND),appending
                childFunction,coefs,hChild,vChild=self.optimizer(appending,parentFunction)
                stepCost=self.actionCost(vParent,vChild)
                gChild=gParent+stepCost
                print 'h=%.2f, g=%.2f , f=%.2f'%(hChild,gChild,hChild+gChild)
                functionCostCoef=(childFunction,hChild,gChild,vChild,coefs)
                successors.append(functionCostCoef)
        return successors




    def optimizer(self,appending,parentFunction):
        def target(*hyper):
            #print 'target called'
            hyper=hyper[0]
            if (self.appending[0]=='composite'):
                appendingHyper=[[hyper[2*i],hyper[2*i+1]] for i in range(2)]
                appendingFunction=['composite']+[ [self.appending[i+1],appendingHyper[i]] for i in range(2) ]
                for i in range(2):    
                    if self.appending[i+1]=='x' and appendingHyper[i][1]>10:
                        appendingHyper[i][1]=np.random.rand()*10
                        
            else:
                appendingHyper=[[hyper[2*i],hyper[2*i+1]] for i in range(len(self.appending))]
                appendingFunction=[ [self.appending[i],appendingHyper[i]] for i in range(len(self.appending)) ]
                for i in range(len(self.appending)):    
                    if self.appending[i]=='x' and appendingHyper[i][1]>10:
                        appendingHyper[i][1]=np.random.rand()*10
                
            childFunction=self.parentFunction+[appendingFunction]            
            xTrain=self.xTrain#[i]  for i in self.randomIndices]
            yTrain=self.yTrain#[i]  for i in self.randomIndices]
            
            kerMat_last_column=[[kerVal(x,appendingFunction) for x in self.xData]]
            kerMat=np.transpose(np.append(np.transpose(self.kerMat),kerMat_last_column,0))
            
            
            kerMatInv=np.linalg.pinv(kerMat[:len(yTrain)])          
            self.coefs=np.dot(kerMatInv,yTrain)
            cost= self.heuristic(kerMat,self.coefs)
            self.childFunction=childFunction
            self.mat=kerMat
            return cost
        #print 'optimizer Called'
        self.randomIndices=np.random.randint(self.xTrain[0],self.xTrain[-1],int(.5*len(self.xTrain)))
        x=np.append(self.xTrain,self.xValid)
        y=np.append(self.yTrain,self.yValid)
        
        self.parentFunction=parentFunction
        self.appending=appending
        leastCostSeen=1000000
        for iteration in range(4):
            if appending[0]=='composite':
                initHyper=2*(np.random.rand(4)-.5)
            else:
                initHyper=2*(np.random.rand(2*len(appending))-.5)
            option={'maxiter':10 ,'eps':1e-1,'disp':False}
            result=scipy.optimize.minimize(target,initHyper, method='L-BFGS-B',jac=None,options=option)
            cost=result.fun
            if cost<leastCostSeen:
                leastCostSeen=cost
                hChild=cost
                childFunction=self.childFunction
                coefs=self.coefs
                deviate=self.prediction(self.mat[:len(y)],coefs,x)-y
        return childFunction,coefs,hChild,deviate

    def actionCost(self,vParent,vChild):
        #cosTheta=np.dot(vParent,vChild)/(hParent*hChild)
        #sinTheta=np.sqrt(abs(1-cosTheta**2))
        #((hChild*sinTheta)**2+(hParent-hChild*cosTheta)**2)**0.5
        return np.linalg.norm(vChild-vParent)
    
    def heuristic(self,kerMat,coefs):
        #heuristic=(1-self.trainSetTrust)*self.jTrain(bases,coefs)+(self.trainSetTrust)*self.jValid(bases,coefs)
        x=np.append(self.xTrain,self.xValid)
        y=np.append(self.yTrain,self.yValid)
        kerMat=kerMat[:len(x)]
        predict=self.prediction(kerMat,coefs,x)
        deviate=predict-y
        #validPredict=self.prediction(bases,coefs,self.xValid)
        #validDeviate=np.dot(validPredict-self.yValid,self.trainSetTrust)
        #deviate=trainDeviate+validDeviate
        return np.linalg.norm(deviate)

        #return heuristic
        
    '''    
    def jTrain(self,bases,coefs):
        trainPredict=self.prediction(bases,coefs,self.xTrain)
        deviate=trainPredict-self.yTrain
        return np.sqrt(np.mean([v**2 for v in deviate]))


    def jValid(self,bases,coefs):
        validPredict=self.prediction(bases,coefs,self.xValid)
        deviate=validPredict-self.yValid
        return np.sqrt(np.mean([v**2 for v in deviate]))
    '''
    def prediction(self,kerMat,coefs,xData):
        return [sum([coefs[j]*kerMat[i][j] for j in range(len(kerMat[0]))]) for i in range(len(kerMat))]

    
    def plot(self,xContinuous,predict,clcTime,pathLength):
        fig=plt.figure()
        for i in range(len(predict)):
            plt.plot(xContinuous,predict[i])
            
        plt.plot(self.xTrain,self.yTrain,'r.')
        plt.plot(self.xValid,self.yValid,'g.')
        plt.plot(self.xTest,self.yTest,'k.')
#        runSet=['run%i'%(i+1) for i in range(runs)]
        lengths=['len=%i'%(pathLength[i]) for i in range(len(pathLength))]
        plt.legend(lengths+['training points','validation points','test points'],prop={'size': 6})
        plt.title('branching=%i,pathLength=%i,clcTimePerRun=%.2f s'%(self.branching,self.pathLength,clcTime))
        plt.savefig('/nfshome/ntatari/BaseSearch/Sep18/%s'%(strftime("%Y-%m-%d %H:%M:%S", gmtime())),dpi=300)
        plt.show()        
        return fig
        
    def train(self):
        return self.treeSearch()

 
def kerVal(x,base):
    if base[0]==['one']:
        a=1
    if base[0]==['composite']:
        a=baseValue(base,1,baseValue(base,2,x))
    else:
        a=1
        for n in range(len(base)):
            a*=baseValue(base,n,x)
    return a



def baseValue(kernel,k,x):
    a=1
    if kernel[k][0]=='x':
        #if abs(float(kernel[k][1][1]))>6:
          #  kernel[k][1][1]=6
        a=((0.001+abs(x+kernel[k][1][0])))**kernel[k][1][1]
    elif kernel[k][0]=='exp':
        #print 'decay,x',kernel[k][1],x
        a=np.exp(kernel[k][1][0]*x)
        #print a
    elif kernel[k][0]=='log':
        a=np.log(abs(kernel[k][1][0]*x+kernel[k][1][1])+0.001)
    elif kernel[k][0]=='log2':
        a=np.log(abs(kernel[k][1][0]*x+kernel[k][1][1]))**2
    elif kernel[k][0]=='log3':
        a=np.log(abs(kernel[k][1][0]*x+kernel[k][1][1]))**3
    elif kernel[k][0]=='log4':
        a=np.log(abs(kernel[k][1][0]*x+kernel[k][1][1]))**4
    elif kernel[k][0]=='cos':
        a=np.cos(kernel[k][1][0]*x+kernel[k][1][1])
    elif kernel[k][0]=='cos2':
        a=np.cos(kernel[k][1][0]*x+kernel[k][1][1])**2
    elif kernel[k][0]=='cos3':
        a=np.cos(kernel[k][1][0]*x+kernel[k][1][1])**3
    elif kernel[k][0]=='cos4':
        a=np.cos(kernel[k][1][0]*x+kernel[k][1][1])**4
    elif kernel[k][0]=='cosh':
        a=np.cosh(kernel[k][1][0]*x+kernel[k][1][1])
    elif kernel[k][0]=='cosh2':
        a=np.cosh(kernel[k][1][0]*x+kernel[k][1][1])**2
    elif kernel[k][0]=='sin':
        a=np.sin(kernel[k][1][0]*x+kernel[k][1][1])
    elif kernel[k][0]=='sin2':
        a=np.sin(kernel[k][1][0]*x+kernel[k][1][1])**2
    elif kernel[k][0]=='sin3':
        a=np.sin(kernel[k][1][0]*x+kernel[k][1][1])**3
    elif kernel[k][0]=='sin4':
        a=np.sin(kernel[k][1][0]*x+kernel[k][1][1])**4 
    elif kernel[k][0]=='sinh':
        a=np.sinh(kernel[k][1][0]*x+kernel[k][1][1])
    elif kernel[k][0]=='sinh2':
        a=np.sinh(kernel[k][1][0]*x+kernel[k][1][1])**2
    elif kernel[k][0]=='tanh':
        a=np.tanh(kernel[k][1][0]*x+kernel[k][1][1])
    elif kernel[k][0]=='tanh2':
        a=np.tanh(kernel[k][1][0]*x+kernel[k][1][1])**2
    elif kernel[k][0]=='Lorenz':
        a=1/(1+(kernel[k][1][0]*x+kernel[k][1][1])**2)
    elif kernel[k][0]=='Gauss':
        a=np.exp(-((x-kernel[k][1][0])**2)/(2*(kernel[k][1][1])**2))
    return a



def randomBase(branching):    
    kernels={}
    functions= ['log','cos','cos2','sin','sin2','tanh','Gauss']#,'exp']#,'x'

    for i in range(branching):
        h=random.choice([1,2,4])
        
        if h==1:
            k=random.choice(functions)        
            kernels[i]=[k]

        elif h==2:
            k1,k2=random.choice(functions),random.choice(functions)
            kernels[i]=[k1,k2]

        elif h==3:
            k1,k2,k3=random.choice(functions),random.choice(functions),random.choice(functions)
            kernels[i]=[k1,k2,k3]

        elif h==4:
            k1,k2=random.choice(functions),random.choice(functions)
            kernels[i]=['composite',k1,k2]
    return kernels
    
    
    
    
    
    
    
    
    
class Color:
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
