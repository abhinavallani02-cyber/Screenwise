# Part 3 — Sreel712: Demo, age analysis and visualization

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

# Visualize the trade-off for the selected threshold.
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(['Missed cases (FN)', 'False alarms (FP)'],
       [comparison.iloc[1]['Missed cases (FN)'], comparison.iloc[1]['False alarms (FP)']],
       color=['#d49a64', '#7b9d87'])
ax.set(ylabel='Number of test records', title=f'Errors at threshold {threshold:.2f}')
plt.tight_layout()
plt.show()

save_project_outputs(record_number, threshold, input_row, score, flag,
                     actual, comparison, subgroups)
print('Saved Screenwise_project_writeup.txt and Screenwise_public_test_evidence.csv.')
print('Open the Colab Files sidebar to download your results.')
print('All three parts are complete.')
```
