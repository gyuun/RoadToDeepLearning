import numpy as np
class perceptron :
    def __init__ (self, x1, x2):
        self.x = np.array([x1, x2])

    def setAttribute(self, x1, x2):
        self.x[0] = x1
        self.x[1] = x2
    
    def AND(self) :
        w = np.array([0.5, 0.5])
        bias = -0.7
        if(np.sum(self.x*w) + bias > 0) :
            return 1
        else:
            return 0
        
    def NAND(self) :
        w = np.array([-0.5, -0.5])
        bias = 0.7
        if(np.sum(self.x*w) + bias > 0):
            return 1
        else:
            return 0
        
    def OR(self) :
        w = np.array([0.5, 0.5])
        bias = -0.4
        if(np.sum(self.x*w) + bias > 0) :
            return 1
        else :
            return 0
    
    def XOR(self) :
        s1 = self.NAND()
        s2 = self.OR()
        new_neuron = perceptron(s1, s2)
        return new_neuron.AND()

p = perceptron(0,0)
attrList = [[0,0], [0,1], [1,0], [1,1]]
for i in np.arange(4):
    print(attrList[i],'\'s And is',p.AND(), 'NAND is',p.NAND(),'OR is', p.OR(),'XOR is' ,p.XOR())
    if i<3:
        x1 = attrList[i+1][0]
        x2 = attrList[i+1][1]
        p.setAttribute(x1,x2)
    

  

