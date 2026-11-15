# Part 3 — Sreel712: Demo, original question and write-up

Open one shared Google Colab notebook. Add three CODE cells. Sreel712 should paste the complete code below into cell 3. Run cells 1, 2 and 3 in that order in the same runtime. Do not paste Markdown fences or document instructions into a code cell. Use a standard Python CPU runtime; the normal Colab libraries are sufficient. An internet connection is required to download the public dataset. Each participant should understand and adapt their assigned part; follow the competition’s AI-assistance rules.

```python
# PART 3 — Sreel712: demo, original question and project write-up
# Paste into the THIRD code cell. Run Parts 1 and 2 first.
# Change the two values below and rerun this cell to explore.
record_number = 636 #@param {type:"integer"}
threshold = 0.35 #@param {type:"number"}
assert 'evidence' in globals(), 'Run Parts 1 and 2 first.'
if not 0 <= threshold <= 1:
    raise ValueError('Threshold must be between 0 and 1.')
if record_number not in evidence.record_number.values:
    raise ValueError('Choose a held-out record number from: ' +
                     ', '.join(map(str, evidence.record_number.head(20))))

record = evidence[evidence.record_number == record_number].iloc[0]
# Recompute from actual feature values through the trained pipeline.
input_row = X.loc[[record_number - 1], FEATURES]
score = float(model.predict_proba(input_row)[0, 1])
flag = score >= threshold
actual = int(record.actual_label)
print('SCREENWISE — public-data demonstration')
print(DISCLAIMER)
print(f'Public record: {record_number}')
display(input_row.rename_axis('public_row_index'))
print('Units: glucose = 2-hour OGTT mg/dL; BP = diastolic mm Hg; BMI = kg/m²; age = years.')
print(f'Model score: {score:.1%} | Threshold: {threshold:.2f}')
print('Output:', 'Flag for review' if flag else 'Below flag threshold')
print('Actual dataset label:', 'Positive' if actual else 'Negative')
print('Outcome:', ('True positive' if flag else 'False negative — missed case')
      if actual else ('False positive — false alarm' if flag else 'True negative'))
print('A score is not a validated individual risk estimate or a diagnosis.')
print('A below-threshold score does not rule out diabetes.')

current = (scores >= threshold).astype(int)
comparison = pd.DataFrame([metrics('Always-negative baseline', baseline),
                           metrics('Model at selected threshold', current)])
print('\nTrade-off at your selected threshold:')
display(comparison.set_index('Method'))
print('Changing the threshold reuses the test set: this is exploration, not new validation.')

print('\nOUR QUESTION: Does the model miss more younger or older positive cases?')
subgroups = []
for name, mask in [('Ages 21–39', evidence.age < 40),
                   ('Ages 40+', evidence.age >= 40)]:
    group = evidence.loc[mask]
    positives = group[group.actual_label == 1]
    missed = int((positives.model_score < threshold).sum())
    subgroups.append({'Age group': name, 'Test records': len(group),
                      'Actual positives': len(positives), 'Missed cases': missed,
                      'FN rate (%)': round(100*missed/len(positives), 1)})
display(pd.DataFrame(subgroups).set_index('Age group'))
print('Small exploratory subgroups; differences do not establish fairness or causation.')

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
evidence.to_csv('Screenwise_public_test_evidence.csv', index=False)
print('\nSaved Screenwise_project_writeup.txt and Screenwise_public_test_evidence.csv.')
print('In Colab, open the Files sidebar to download them.')
print('All three parts are complete.')
```
