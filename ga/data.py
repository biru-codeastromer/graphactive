import numpy as np
import networkx as nx

def make_sbm(seed=42,n=1500,k=3,p_in=0.05,p_out=0.005,d=64,imbalance=0.1):
    rng=np.random.RandomState(seed)
    base=n//k
    sizes=[base for _ in range(k)]
    for i in range(n-sum(sizes)):
        sizes[i%k]+=1
    shift=int(imbalance*base)
    for i in range(k):
        if i%2==0:
            sizes[i]=max(10,sizes[i]+shift)
        else:
            sizes[i]=max(10,sizes[i]-shift)
    while sum(sizes)>n:
        for i in range(k):
            if sizes[i]>10 and sum(sizes)>n:
                sizes[i]-=1
    while sum(sizes)<n:
        sizes[0]+=1
    B=np.full((k,k),p_out,dtype=float)
    for i in range(k):
        B[i,i]=p_in
    G=nx.stochastic_block_model(sizes,B,seed=seed)
    A=nx.to_numpy_array(G,dtype=float)
    A=A+np.eye(A.shape[0])
    deg=A.sum(1)
    D=np.diag(1.0/np.sqrt(np.maximum(deg,1e-8)))
    An=D@A@D
    y=[]
    for idx,s in enumerate(sizes):
        y+=[idx]*s
    y=np.array(y,dtype=int)
    rng.shuffle(y)
    mu=rng.randn(k,d)*0.5
    X=np.zeros((n,d),dtype=float)
    for c in range(k):
        idx=np.where(y==c)[0]
        X[idx]=mu[c]+rng.randn(len(idx),d)*1.0
    return An,X,y
