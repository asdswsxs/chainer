from sklearn import datasets
import numpy as np
import chainer
from chainer import cuda,Function,report,training,utils,Variable,iterators,optimizers,serializers,Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

#set data
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0
index = np.arange(N)
xtrain = X[index[index %2 != 0],:]
ytrain = Y2[index[index %2 != 0],:]
xtest  = X[index[index %2 == 0],:]
y_ans  = Y[index[index %2 == 0]]

#Define model

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain,self).__init__(
            l1 = L.Linear(4,6),#4 for data feature ;6 is middle layer Neural Networks
            l2 = L.Linear(6,3),#3 classification
        )

    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x),y)

    def fwd(self,x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

#Initialize model

model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

#learn

for i in range(10000):
    x = Variable(xtrain)
    y = Variable(ytrain)
    model.cleargrads()
    loss = model(x,y)
    loss.backward()
    optimizer.update()

#test

x_train =  Variable(xtest)
yy = model.fwd(x_train)
ans = yy.data
nrow,ncol = ans.shape
done = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print(ans[i,:],cls)
    if cls == y_ans[i]:
        done += 1
print(done,"/",nrow,"=",(done*1.0)/nrow)
#(73, '/', 75, '=', 0.9733333333333334)