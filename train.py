import json, hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
p=Path('diabetes.csv')
if not p.exists():
    from urllib.request import urlretrieve
    urlretrieve('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', p)
Path('dist').mkdir(exist_ok=True)
a=np.loadtxt(p,delimiter=',')
# Routine screening inputs: glucose, diastolic BP, BMI, age.
X=a[:,[1,2,5,7]].copy(); y=a[:,8].astype(int)
X[:,:3]=np.where(X[:,:3]==0,np.nan,X[:,:3])
tr,te=train_test_split(np.arange(len(y)),test_size=.25,random_state=42,stratify=y)
m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
m.fit(X[tr],y[tr]); prob=m.predict_proba(X[te])[:,1]
imp,sc,lr=m.steps[0][1],m.steps[1][1],m.steps[2][1]
out=dict(median=imp.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),test=[dict(id=f'PUBLIC-{i+1:03}',x=[None if np.isnan(v) else float(v) for v in X[i]],y=int(y[i]),score=float(s)) for i,s in zip(te,prob)],auc=roc_auc_score(y[te],prob),train_n=len(tr),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path('dist/model.json').write_text(json.dumps(out))
print(json.dumps({'train':len(tr),'test':len(te),'confusion_at_035':confusion_matrix(y[te],prob>=.35).tolist(),'auc':out['auc']}))
