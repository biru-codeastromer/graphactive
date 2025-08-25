import numpy as np
from .data import make_sbm
from .model import gcn_features, fit_predict

def split_indices(rng,n,test_frac=0.2,val_frac=0.0):
    idx=np.arange(n)
    rng.shuffle(idx)
    n_test=int(n*test_frac)
    test_idx=idx[:n_test]
    pool_idx=idx[n_test:]
    return pool_idx,test_idx

def init_labels(rng,y,initial_per_class=10):
    k=len(np.unique(y))
    labeled=[]
    for c in range(k):
        idx=np.where(y==c)[0]
        rng.shuffle(idx)
        take=min(initial_per_class,len(idx))
        labeled+=list(idx[:take])
    labeled=np.array(sorted(set(labeled)),dtype=int)
    return labeled

def entropy(p):
    q=np.clip(p,1e-9,1-1e-9)
    return -(q*np.log(q)).sum(1)

def round_robin_by_community(rng,communities,order_by,quota,size):
    groups={}
    for i,c in enumerate(communities):
        groups.setdefault(c,[]).append(i)
    for c in groups:
        groups[c]=sorted(groups[c],key=lambda i:order_by[i],reverse=True)
    picks=[]
    counts={c:0 for c in groups}
    keys=list(groups.keys())
    while len(picks)<size:
        keys=sorted(keys,key=lambda c:counts[c]/max(1,quota[c]))
        moved=False
        for c in keys:
            while groups[c] and groups[c][0] in picks:
                groups[c].pop(0)
            if groups[c]:
                picks.append(groups[c].pop(0))
                counts[c]+=1
                moved=True
                if len(picks)>=size:
                    break
        if not moved:
            break
    return np.array(picks,dtype=int)

def acquire(typ,proba,comm,already_selected,batch,lam=0.5,rng=None):
    n=len(proba)
    ent=entropy(proba)
    cand=np.setdiff1d(np.arange(n),already_selected,assume_unique=True)
    if typ=="entropy":
        order=np.argsort(-ent[cand])
        return cand[order[:batch]]
    if typ=="community":
        uniq,counts=np.unique(comm[cand],return_counts=True)
        quota={c:counts[i] for i,c in enumerate(uniq)}
        return cand[round_robin_by_community(rng,comm[cand],ent[cand],quota,batch)]
    if typ=="hybrid":
        selected_comm={}
        picks=[]
        pool=set(cand.tolist())
        ent_n=(ent-ent.min())/(ent.max()-ent.min()+1e-12)
        while len(picks)<batch and pool:
            scores=[]
            ids=list(pool)
            for i in ids:
                d=1.0/(1.0+selected_comm.get(comm[i],0))
                scores.append(lam*ent_n[i]+(1-lam)*d)
            j=ids[int(np.argmax(scores))]
            picks.append(j)
            pool.remove(j)
            selected_comm[comm[j]]=selected_comm.get(comm[j],0)+1
        return np.array(picks,dtype=int)
    return cand[:batch]

def run_active(seed=42,n=1500,k=3,p_in=0.05,p_out=0.005,d=64,initial=10,batch=20,rounds=10,layers=2,lam=0.5):
    rng=np.random.RandomState(seed)
    An,X,y=make_sbm(seed=seed,n=n,k=k,p_in=p_in,p_out=p_out,d=d,imbalance=0.1)
    comm=y.copy()
    pool_idx,test_idx=split_indices(rng,n,test_frac=0.2,val_frac=0.0)
    labeled=init_labels(rng,y[pool_idx],max(1,initial//k))
    labeled=pool_idx[labeled]
    pool=set(pool_idx.tolist())
    for j in labeled.tolist():
        pool.discard(j)
    Z=gcn_features(An,X,layers=layers)
    curves={"entropy":[],"community":[],"hybrid":[]}
    counts=[]
    for r in range(rounds+1):
        for typ in ["entropy","community","hybrid"]:
            L=np.array(sorted(labeled),dtype=int)
            ytr=y[L]
            pred,proba=fit_predict(Z[L],ytr,Z,test_idx[0] if False else seed)
            te_pred,te_proba=fit_predict(Z[L],ytr,Z[test_idx],seed)
            acc=float((te_pred==y[test_idx]).mean())
            curves[typ].append(acc)
        counts.append(len(labeled))
        if r==rounds:
            break
        pred_full,proba_full=fit_predict(Z[np.array(sorted(labeled))],y[np.array(sorted(labeled))],Z,seed)
        proba_pool=proba_full[list(pool)]
        comm_pool=comm[list(pool)]
        already=np.array([],dtype=int)
        pick_e=acquire("entropy",proba_full,comm,already,batch,lam,rng)
        pick_c=acquire("community",proba_full,comm,already,batch,lam,rng)
        pick_h=acquire("hybrid",proba_full,comm,already,batch,lam,rng)
        choose=pick_h
        choose=list(choose)
        rng.shuffle(choose)
        choose=choose[:batch]
        for j in choose:
            if j in pool:
                labeled=np.append(labeled,j)
                pool.discard(j)
    return np.array(counts),np.array(curves["entropy"]),np.array(curves["community"]),np.array(curves["hybrid"])
