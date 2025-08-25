import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def gcn_features(An,X,layers=2):
    Z=X.copy()
    for _ in range(layers):
        Z=An@Z
    return Z

def fit_predict(Xtr,ytr,Xte,seed=42):
    scaler=StandardScaler(with_mean=False)
    Xtr2=scaler.fit_transform(Xtr)
    clf=LogisticRegression(max_iter=1000,random_state=seed,multi_class="auto")
    clf.fit(Xtr2,ytr)
    Xte2=scaler.transform(Xte)
    proba=clf.predict_proba(Xte2)
    pred=proba.argmax(1)
    return pred,proba
