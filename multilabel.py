
import numpy as np

def BinaryRelevance(X,y):
    Y = np.transpose(y)
    #labels = Y.copy()
    for e,t in enumerate(Y):
        #t[t>0]=e+1
        yield X,t
    
def CalibratedLabelRanking(X,y):
    Y = np.transpose(y)
    indices, pairs = [],[]
    for e1,i in enumerate(Y):
        for e2,j in enumerate(Y[e1+1:]):
            index, pair = [],[]
            for x in range(len(i)):
                if i[x] == j[x]:
                    continue
                else:
                    pair.append(i[x]*(e1+1)+j[x]*(e1+e2+2))
                    index.append(x)
            pairs.append(pair[:])
            indices.append(index[:])
            
        index, pair = [],[]
        for e,x in enumerate(i):
            pair.append(x*(e1+1))
            index.append(e)
        pairs.append(pair[:])
        indices.append(index[:])
        
    for e,t in enumerate(pairs):
        data = X[np.array(indices[e])]
        yield data,t
