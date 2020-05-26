
import numpy as np

def BinaryRelevance(X,y):
    Y = np.transpose(y)
    for e,t in enumerate(Y):
        t[t>0]=e+1
        yield X,t
    
def CalibratedLabelRanking(X,y):
    Y = np.transpose(y)
    indices, pairs = [],[]
    for e1,i in enumerate(Y):
        for e2,j in enumerate(Y[e1+1:]):
            index1, pair1 = [],[]
            for x in range(len(i)):
                if i[x] == j[x]:
                    continue
                else:
                    pair1.append(i[x]*(e1+1)+j[x]*(e1+e2+2))
                    index1.append(x)
            pairs.append(pair1)
            indices.append(index1)
            
        index2, pair2 = [],[]
        for e,x in enumerate(i):
            pair2.append(x*(e1+1))
            index2.append(e)
        pairs.append(pair2)
        indices.append(index2)
        
    for e,t in enumerate(pairs):
        data = X[np.array(indices[e])]
        yield data,t
