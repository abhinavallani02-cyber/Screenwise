# Part 1 — Abhinav: Public data and preparation

Open one shared Google Colab notebook. Add three CODE cells. Abhinav should paste the complete code below into cell 1. Run cells 1, 2 and 3 in that order in the same runtime. Do not paste Markdown fences or document instructions into a code cell. Use a standard Python CPU runtime; the normal Colab libraries are sufficient. An internet connection is required to download the public dataset. Each participant should understand and adapt their assigned part; follow the competition’s AI-assistance rules.

```python
# PART 1 — Abhinav: public data and preparation
# Paste into the FIRST code cell, then run it.
from io import StringIO
from urllib.request import urlopen
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

DISCLAIMER = 'This project is a prototype and is not medical advice.'
print('SCREENWISE — AI + Health')
print(DISCLAIMER)
print('User: Ana Martinez, a fictional community health nurse in Phoenix.')
print('Moment: reviewing screening follow-ups after a busy clinic session.')
print('Goal: explore which public records a model flags for closer review.')
print('This demonstration must not be used to prioritize actual patients.')

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw = urlopen(URL, timeout=30).read()
columns = ['pregnancies', 'glucose', 'diastolic_bp', 'skin_thickness',
           'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv(StringIO(raw.decode()), header=None, names=columns)
assert data.shape == (768, 9), 'Unexpected dataset shape; check the source.'
assert set(data['label']) == {0, 1}
FEATURES = ['glucose', 'diastolic_bp', 'bmi', 'age']
X = data[FEATURES].copy()
# Zero values in these measurements mean missing data, not healthy values.
X[['glucose', 'diastolic_bp', 'bmi']] = X[['glucose', 'diastolic_bp', 'bmi']].replace(0, np.nan)
y = data['label'].astype(int)
train_ids, test_ids = train_test_split(
    np.arange(len(data)), test_size=0.25, random_state=42, stratify=y)
X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
assert set(train_ids).isdisjoint(test_ids)
print(f'Public records: {len(data)} | Train: {len(train_ids)} | Test: {len(test_ids)}')
print('Dataset SHA-256:', hashlib.sha256(raw).hexdigest())
print('Source:', URL)
print('Historical Pima Indians Diabetes dataset, originally attributed to NIDDK.')
print('Population: women aged 21+ of Pima heritage; not representative of everyone.')
print('Target: existing dataset diabetes label, NOT future disease onset.')
display(X_test.head().rename_axis('public_row_index'))
print('Part 1 complete. Run Part 2 next.')
```
