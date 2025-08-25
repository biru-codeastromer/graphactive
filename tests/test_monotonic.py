import numpy as np
from ga.active import run_active

def test_final_ge_initial_and_hybrid_lead():
    counts,ent,com,hyb=run_active(seed=7,n=900,k=3,p_in=0.06,p_out=0.004,d=48,initial=9,batch=15,rounds=8,layers=2,lam=0.6)
    assert hyb[-1]>=hyb[0]-1e-6
    assert ent[-1]>=ent[0]-1e-6
    early_hyb=np.trapz(hyb[:len(hyb)//2],counts[:len(counts)//2])
    early_ent=np.trapz(ent[:len(ent)//2],counts[:len(counts)//2])
    assert early_hyb>=early_ent-1e-6
