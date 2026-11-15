# Part 2 — Nehanth185: Short evaluation and verification

Open one shared Google Colab notebook. Add three CODE cells. Nehanth185 should paste the complete code below into cell 2. Run cells 1, 2 and 3 in that order in the same runtime. Do not paste Markdown fences or document instructions into a code cell. Use a standard Python CPU runtime; the normal Colab libraries are sufficient. An internet connection is required to download the public dataset. Each participant should understand and adapt their assigned part; follow the competition’s AI-assistance rules.

```python
# PART 2 — Nehanth185: short evaluation and verification
# Paste into the SECOND code cell after running Part 1.
assert 'model' in globals(), 'Run Part 1 first.'
print('MODEL VS SIMPLE BASELINE — threshold 0.35')
display(results.set_index('Method'))
print(f'Model ROC AUC: {roc_auc_score(y_test, scores):.4f}; baseline AUC: 0.5000')
print('Missed cases (FN) are actual positives predicted negative.')
print('False alarms (FP) are actual negatives predicted positive.')
print('FN rate uses actual positives; FP rate uses actual negatives.')
print('Accuracy alone hides the baseline’s 67 missed positives.')

# Check that the test records are separate and every prediction is counted.
assert set(train_ids).isdisjoint(test_ids)
assert len(scores) == len(y_test) == 192
assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
assert int(cm.sum()) == len(y_test)
print('Verification passed. Run Part 3 for the demo and original question.')
```
