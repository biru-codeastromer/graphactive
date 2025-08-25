import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ga.active import run_active

st.set_page_config(page_title="GraphActive",page_icon="🧭",layout="centered")

seed=st.sidebar.number_input("seed",value=42,step=1)
n=st.sidebar.number_input("nodes",min_value=100,value=1500,step=100)
k=st.sidebar.number_input("communities",min_value=2,value=3,step=1)
p_in=st.sidebar.slider("p_in",min_value=0.01,max_value=0.2,value=0.05,step=0.01)
p_out=st.sidebar.slider("p_out",min_value=0.001,max_value=0.05,value=0.005,step=0.001)
d=st.sidebar.number_input("feat dim",min_value=4,value=64,step=16)
initial=st.sidebar.number_input("initial labels",min_value=1,value=10,step=5)
batch=st.sidebar.number_input("batch size",min_value=1,value=20,step=5)
rounds=st.sidebar.number_input("rounds",min_value=1,value=10,step=1)
layers=st.sidebar.number_input("gcn layers",min_value=1,value=2,step=1)
lam=st.sidebar.slider("hybrid lambda",min_value=0.0,max_value=1.0,value=0.5,step=0.05)

if st.button("run"):
    counts,ent,com,hyb=run_active(seed=int(seed),n=int(n),k=int(k),p_in=float(p_in),p_out=float(p_out),d=int(d),initial=int(initial),batch=int(batch),rounds=int(rounds),layers=int(layers),lam=float(lam))
    st.subheader("curves")
    fig=plt.figure()
    plt.plot(counts,ent,label="uncertainty")
    plt.plot(counts,com,label="community-aware")
    plt.plot(counts,hyb,label="hybrid")
    plt.xlabel("labeled count")
    plt.ylabel("test accuracy")
    plt.legend()
    plt.title("active learning on graphs")
    st.pyplot(fig)
    auc_ent=float(np.trapz(ent,counts))
    auc_com=float(np.trapz(com,counts))
    auc_hyb=float(np.trapz(hyb,counts))
    st.subheader("metrics")
    st.json({"auc_entropy":auc_ent,"auc_community":auc_com,"auc_hybrid":auc_hyb,"early_advantage":float(np.trapz(hyb[:len(hyb)//2],counts[:len(counts)//2])-np.trapz(ent[:len(ent)//2],counts[:len(counts)//2]))})
else:
    st.info("set parameters and click run")
