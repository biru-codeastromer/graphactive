import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from .active import run_active

p=argparse.ArgumentParser()
p.add_argument("cmd",choices=["reproduce","plot"])
p.add_argument("--seed",type=int,default=42)
p.add_argument("--n",type=int,default=1500)
p.add_argument("--k",type=int,default=3)
p.add_argument("--p_in",type=float,default=0.05)
p.add_argument("--p_out",type=float,default=0.005)
p.add_argument("--d",type=int,default=64)
p.add_argument("--initial",type=int,default=10)
p.add_argument("--batch",type=int,default=20)
p.add_argument("--rounds",type=int,default=10)
p.add_argument("--layers",type=int,default=2)
p.add_argument("--lam",type=float,default=0.5)
a=p.parse_args()

counts,ent,com,hyb=run_active(seed=a.seed,n=a.n,k=a.k,p_in=a.p_in,p_out=a.p_out,d=a.d,initial=a.initial,batch=a.batch,rounds=a.rounds,layers=a.layers,lam=a.lam)
auc_ent=float(np.trapz(ent,counts))
auc_com=float(np.trapz(com,counts))
auc_hyb=float(np.trapz(hyb,counts))

if a.cmd=="reproduce":
    out={"counts":counts.tolist(),"auc_entropy":auc_ent,"auc_community":auc_com,"auc_hybrid":auc_hyb,"early_advantage":float(np.trapz(hyb[:len(hyb)//2],counts[:len(counts)//2])-np.trapz(ent[:len(ent)//2],counts[:len(counts)//2]))}
    print(json.dumps(out,indent=2))
else:
    os.makedirs("outputs",exist_ok=True)
    fig=plt.figure()
    plt.plot(counts,ent,label="uncertainty")
    plt.plot(counts,com,label="community-aware")
    plt.plot(counts,hyb,label="hybrid")
    plt.xlabel("labeled count")
    plt.ylabel("test accuracy")
    plt.legend()
    plt.title("active learning on graphs")
    fig.savefig("outputs/accuracy_curves.png",bbox_inches="tight",dpi=160)
    print(json.dumps({"auc_entropy":auc_ent,"auc_community":auc_com,"auc_hybrid":auc_hyb},indent=2))
