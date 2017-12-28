import numpy
import matplotlib.pyplot

class optimizers(object):
    def __init__(self,inp,fx,dfx,lr):
        assert(type(inp)==numpy.ndarray),"Input shoud be numpy array"
        self.input=inp
        self.lr=lr
        self.fx=fx
        self.dfx=dfx
        self.loss=[]
    def plot(self):
        assert(len(self.loss)>0),"Nothing to plot"
        return matplotlib.pyplot.plot(self.loss)


class SGD(optimizers):
    def __init__(self,inp,fx,dfx,lr):
        super(SGD,self).__init__(inp,fx,dfx,lr)
    def step(self):
        self.loss.append(sum(self.fx(self.input)))
        self.input = self.input - self.lr*self.dfx(self.input)
        return

class momentum(optimizers):
    def __init__(self,inp,fx,dfx,lr):
        super(momentum,self).__init__(inp,fx,dfx,lr)
        self.m=numpy.zeros_like(inp)
    def step(self):
        self.loss.append(sum(self.fx(self.input)))
        self.m=0.9*self.m-self.lr*self.dfx(self.input)
        self.input =self.input + self.m


class nestrov_momentum(optimizers):
    def __init__(self,inp,fx,dfx,lr):
        super(nestrov_momentum,self).__init__(inp,fx,dfx,lr)
        self.m=numpy.zeros_like(inp)
    def step(self):
        self.loss.append(sum(self.fx(self.input)))
        v=0.9*self.m-self.lr*self.dfx(self.input)
        self.m=0.9*(0.9*self.m-self.lr*self.dfx(self.input))
        self.input =self.input + self.m
        return

class Adagrad(optimizers):
    def __init__(self,inp,fx,dfx,lr):
        super(Adagrad,self).__init__(inp,fx,dfx,lr)
        self.chache=0.0
    def step(self):
        current_deriv=self.dfx(self.input)
        self.loss.append(sum(self.fx(self.input)))
        self.chache = self.chache + sum(current_deriv**2)
        lr=(self.lr)/(numpy.sqrt(self.chache)+1e-8)
        self.input= self.input - lr*current_deriv
        return

class Adam(optimizers):
    def __init__(self,inp,fx,dfx,lr):
        super(Adam,self).__init__(inp,fx,dfx,lr)
        self.v= numpy.zeros_like(inp)
        self.m=numpy.zeros_like(inp)
    def step(self):
        self.loss.append(sum(self.fx(self.input)))
        gt=self.dfx(self.input)
        self.m=self.m*0.9 + (0.1)*gt
        self.v=self.v*0.99 + 0.01*gt
        mt=self.m / (0.1)
        vt=self.v / (0.01)
        self.input =self.input -((self.lr *mt )/(numpy.sqrt(vt )+ 1e-8))
        return
