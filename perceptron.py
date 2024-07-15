import numpy as np
class perceptronLogic :
    def __init__ (self, x):
        # x is nd.array()
        self.x = np.append(x, 1)

    def setAttribute(self, x):
        self.x = np.append(x, 1)

    def result(self, w):
        if(np.sum(np.dot(self.x, w)) > 0) :
            return 1
        else:
            return 0
    
    def AND(self) :
        w = np.array([0.5, 0.5, -0.7])
        return self.result(w)
        
    def NAND(self) :
        w = np.array([-0.5, -0.5, 0.7])
        return self.result(w)
        
    def OR(self) :
        w = np.array([0.5, 0.5, -0.3])
        return self.result(w)
    
    def XOR(self) :
        s1 = self.NAND()
        s2 = self.OR()
        new_neuron = perceptronLogic(np.array([s1, s2]))
        return new_neuron.AND()
"""
attrList = [[0,0], [0,1], [1,0], [1,1]]
attrArr = np.array(attrList)
print(attrArr[0])
p = perceptronLogic(attrArr[0])

for i in np.arange(4):
    print(attrList[i],'\'s And is',p.AND(), 'NAND is',p.NAND(),'OR is', p.OR(),'XOR is', p.XOR())
    if i<3:
        p.setAttribute(attrArr[i+1])

"""

  

