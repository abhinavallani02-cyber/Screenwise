# Part 1 — Abhinav: Main implementation: data, model and shared functions

Open one shared Google Colab notebook. Add three CODE cells. Abhinav should paste the complete code below into cell 1. Run cells 1, 2 and 3 in that order in the same runtime. Do not paste Markdown fences or document instructions into a code cell. Use a standard Python CPU runtime; the normal Colab libraries are sufficient. An internet connection is required to download the public dataset. Each participant should understand and adapt their assigned part; follow the competition’s AI-assistance rules.

```python
# PART 1 — Abhi: main implementation — data, model and shared functions
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


# Train the model and prepare reusable evaluation data.
model = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42))
# All preprocessing learns from training data only.
model.fit(X_train, y_train)
scores = model.predict_proba(X_test)[:, 1]
majority_label = int(y_train.mode().iloc[0])
baseline = np.full(len(y_test), majority_label)
THRESHOLD = 0.35  # Illustrative, not clinically chosen or validated.

def metrics(name, predictions):
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {'Method': name, 'Accuracy (%)': round(100*(tp+tn)/len(y_test), 2),
            'True positives': int(tp), 'True negatives': int(tn),
            'Missed cases (FN)': int(fn), 'False alarms (FP)': int(fp),
            'FN rate (%)': round(100*fn/(tp+fn), 2),
            'FP rate (%)': round(100*fp/(tn+fp), 2)}

predictions = (scores >= THRESHOLD).astype(int)
results = pd.DataFrame([metrics('Always-negative baseline', baseline),
                        metrics('Logistic regression', predictions)])





evidence = X_test.copy()
evidence.insert(0, 'record_number', test_ids + 1)
evidence['actual_label'] = y_test.to_numpy()
evidence['model_score'] = scores
evidence['predicted_label'] = predictions
errors = evidence[evidence.actual_label != evidence.predicted_label]


def save_project_outputs(record_number, threshold, input_row, score, flag, actual, comparison, subgroups):
    report = f'''SCREENWISE — AI + HEALTH
    User: Ana Martinez, a fictional community health nurse in a Phoenix teaching-clinic scenario.
    Moment: reviewing screening follow-ups with limited time. This is an educational simulation.
    Inputs: glucose, diastolic blood pressure, BMI, age. Output: model score and review flag.
    Data: public Pima Indians Diabetes dataset only; 768 records. No private patient records.
    Source: {URL}
    Method: stratified 576/192 train/test split, seed 42; training-only median imputation and
    standardization; logistic regression. Zero glucose, BP and BMI values are treated as missing.
    The target is the existing diabetes label, not future onset.
    
    Results at selected threshold {threshold:.2f}:
    {comparison.to_string(index=False)}
    Model ROC AUC: {roc_auc_score(y_test, scores):.4f}; baseline AUC: 0.5000.
    
    Real input example: record {record_number}, {input_row.iloc[0].to_dict()}.
    Score: {score:.6f}, predicted label: {int(flag)}, actual label: {actual}.
    Failure example at the original threshold 0.35: public record 636 has glucose 104,
    BP 72, BMI 31.2, age 38 and score about 0.1774. Its actual label is positive,
    but the model does not flag it. Overlap between classes and omitted clinical factors
    may contribute; the specific cause is unknown.
    Original question: do missed-positive rates differ for ages 21–39 and 40+?
    {pd.DataFrame(subgroups).to_string(index=False)}
    
    Limits: historical data restricted to women aged 21+ of Pima heritage. No validation
    for other populations, no external or prospective evaluation, and no calibration study.
    Threshold 0.35 and the age split are illustrative. Exploring this test set does not
    provide independent threshold validation. Small subgroup differences are uncertain.
    Next: test a locked model and threshold on a compatible, independent public dataset.
    This project is a prototype and is not medical advice.
    '''
    with open('Screenwise_project_writeup.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    export = evidence.copy()
    export['predicted_label'] = (export.model_score >= threshold).astype(int)
    export['threshold'] = threshold
    export.to_csv('Screenwise_public_test_evidence.csv', index=False)

print('Main implementation ready. Run Part 2 next.')
```
