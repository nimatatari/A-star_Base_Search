import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return (0.3*np.sin(2*x)*(1+(x-5)**2)+0.2*error(1))+ np.exp(-((x-4)**2)/5)
    
def error(var):
    return np.random.normal(0,var)

xData=np.linspace(1,10,50)
yData=[function(x) for x in xData]
z=zip(xData,yData)
print z
np.save('data/artData.npy',z)
print 'dataSD=',np.sqrt(np.var(yData))


trainSetRatio=0.3#srvm.trainSetRatio
validationSetRatio=.6#srvm.validationSetRatio


trainSetLen=int(trainSetRatio*len(xData))
validationSetLen=int(validationSetRatio*(len(xData)-trainSetLen))


xTrain=xData[:trainSetLen]
yTrain=yData[:trainSetLen]
xValid=xData[trainSetLen:trainSetLen+validationSetLen]
yValid=yData[trainSetLen:trainSetLen+validationSetLen]
xTest=xData[trainSetLen+validationSetLen:]
yTest=yData[trainSetLen+validationSetLen:]


plt.plot(xTrain,yTrain,'r.')
plt.plot(xValid,yValid,'g.')
plt.plot(xTest,yTest,'k.')
plt.show()

